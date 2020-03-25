#! -*- coding: utf-8 -*-

from PyTorch.UniLM import UniLMModel
from PyTorch.tokenizer import SimpleTokenizer,load_vocab
from PyTorch.configuration_bert import BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import os,json,codecs,logging,random
from tqdm import trange,tqdm
import numpy as np
import argparse

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def read_text(max_input_len,max_output_len):
    df = pd.read_csv('../data/train.csv')
    text = df['text'].values
    summarization = df['summarization'].values

    for t, s in zip(text, summarization):
        if len(s) <= max_output_len:
            yield t[:max_input_len], s

def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def data_generator(tokenizer,batch_size=4,max_output_len=32,max_input_len=450):
    while True:
        X, S = [], []
        for a, b in read_text(max_input_len,max_output_len):
            # x为text和summaryzation融合信息，s为融合后的segment ids
            x, s = tokenizer.encode(a, b)
            X.append(x)
            S.append(s)
            if len(X) == batch_size:
                X = padding(X)
                S = padding(S)
                yield [X, S], None
                X, S = [], []

def gen_sent(s,tokenizer,model,args):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    topk = args.topk
    token_ids, segment_ids = tokenizer.encode(s[:args.max_input_len])
    # 候选答案id
    target_ids = [[] for _ in range(topk)]
    # 候选答案分数
    target_scores = [0] * topk
    # 强制要求输出不超过max_output_len字
    model.eval()

    with torch.no_grad():
        for i in range(args.max_output_len):
            _target_ids = [token_ids + t for t in target_ids]
            _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]

            input_ids = torch.tensor(_target_ids, dtype=torch.long).to(args.device)
            token_type_ids = torch.tensor(_segment_ids, dtype=torch.long).to(args.device)

            outputs = model(input_ids, catcu_lss=False, token_type_ids=token_type_ids)

            _probas = outputs[0][:, -1, :]
            # 取对数，方便计算
            # _log_probas = np.log(_probas + 1e-6)
            _log_probas = _probas.numpy()
            # 每一项选出topk
            _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]
            _candidate_ids, _candidate_scores = [], []
            for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
                # 预测第一个字的时候，输入的topk事实上都是同一个，
                # 所以只需要看第一个，不需要遍历后面的。
                if i == 0 and j > 0:
                    continue
                for k in _topk_arg[j]:
                    _candidate_ids.append(ids + [k])
                    _candidate_scores.append(sco + _log_probas[j][k])
            _topk_arg = np.argsort(_candidate_scores)[-topk:]
            for j, k in enumerate(_topk_arg):
                # target_ids[j].append(_candidate_ids[k][-1])
                target_ids[j] = _candidate_ids[k]
                target_scores[j] = _candidate_scores[k]
            ends = [j for j, k in enumerate(target_ids) if k[-1] == 100]
            if len(ends) > 0:
                k = np.argmax([target_scores[j] for j in ends])
                return tokenizer.decode(target_ids[ends[k]])

    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def main():
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--data_dir",default='../data/train.csv',type=str,required=False,)
    parser.add_argument("--model_name_or_path",default='/Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch',type=str,required=False,)
    parser.add_argument("--output_dir",default='./modles',type=str,required=False,)

    # Other parameters
    parser.add_argument("--cache_dir",default="",type=str,)
    parser.add_argument("--max_input_len",default=450,type=int,help="文本最长输入长度")
    parser.add_argument("--max_output_len", default=32, type=int, help="最长输出摘要长度")
    parser.add_argument("--cut_vocab", default=True, action="store_true", help="是否精简原字表")
    parser.add_argument("--min_count", default=30, type=int, help="精简掉出现频率少于此的word")
    parser.add_argument("--topk", default=4, type=int, help="beam search参数")
    parser.add_argument("--topp", default=0., type=float, help="核采样参数")
    parser.add_argument("--do_train",default=True, action="store_true", help="是否fine tuning")
    parser.add_argument("--do_show",default=True, action="store_true", )
    parser.add_argument("--batch_size", default=4, type=int,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="学习率衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁减值")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练epochs次数",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="学习率线性预热步数")
    parser.add_argument("--logging_steps", type=int, default=500, help="每多少步打印日志")
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    parser.add_argument("--max_steps",default=200000,type=int,help="训练的总步数",)
    parser.add_argument("--save_steps", default=50000, type=int, help="保存的间隔steps", )

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir) and os.listdir(args.output_dir)
        and args.do_train and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    # 建立分词器
    _token_dict = load_vocab(os.path.join(args.model_name_or_path, 'vocab.txt'))
    # keep_words是在bert中保留的字表
    token_dict, keep_words = {}, []
    if args.cut_vocab:
        if os.path.exists('./seq2seq_config.json'):
            chars = json.load(open('./seq2seq_config.json', encoding='utf-8'))
        else:
            chars = {}
            for a in tqdm(read_text(args.max_input_len, args.max_output_len), desc='构建字表中'):
                for b in a:
                    for w in b:
                        chars[w] = chars.get(w, 0) + 1
            chars = [(i, j) for i, j in chars.items() if j >= args.min_count]
            chars = sorted(chars, key=lambda c: - c[1])
            chars = [c[0] for c in chars]
            json.dump(
                chars,
                codecs.open('./seq2seq_config.json', 'w', encoding='utf-8'),
                indent=4,
                ensure_ascii=False
            )
        for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])

        for c in chars:
            if c in _token_dict:
                token_dict[c] = len(token_dict)
                keep_words.append(_token_dict[c])

    tokenizer = SimpleTokenizer(token_dict if args.cut_vocab else _token_dict)

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=[0,1],
        finetuning_task='unilm',
        cache_dir=None,
    )
    config.keep_words = keep_words
    model = UniLMModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=None,
    )
    model.to(args.device)
    # 精简词表
    if args.cut_vocab:
        model.resize_token_embeddings(new_num_tokens=len(keep_words),keep_words=keep_words)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        t_total = args.max_steps
        tb_writer = SummaryWriter('./tensorboardX')
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,eps=args.adam_epsilon)

        train_epochs = trange(args.num_train_epochs, desc='开始训练epoch')
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        for epoch in train_epochs:
            for step, batch in enumerate(data_generator(tokenizer,batch_size=args.batch_size,
                                                        max_output_len=args.max_output_len,max_input_len=args.max_input_len)):
                model.train()

                input_ids = torch.tensor(batch[0][0], dtype=torch.long).to(device)
                token_type_ids = torch.tensor(batch[0][1], dtype=torch.long).to(device)

                outputs = model(input_ids, token_type_ids=token_type_ids)
                loss = outputs[0]

                # 只计算摘要部分的输出loss
                y_mask = token_type_ids[:, 1:].reshape(-1).contiguous()
                loss = torch.sum(loss * y_mask) / torch.sum(y_mask)

                loss.backward()
                tr_loss += loss.item()
                if (step+1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)  # 梯度截断

                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        logs["loss损失"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        logger.info("Saving optimizer states to %s", output_dir)

                if args.max_steps > 0 and global_step > t_total:
                    break

            if args.max_steps > 0 and global_step > t_total:
                train_epochs.close()
                break
        tb_writer.close()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss/global_step)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", args.output_dir)

        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        logger.info("Saving optimizer states to %s", args.output_dir)

    if args.do_show:
        # config = BertConfig.from_pretrained(
        #     args.output_dir,
        #     num_labels=[0, 1],
        #     finetuning_task='unilm',
        #     cache_dir=None,
        # )
        # # config.keep_words = keep_words
        # # config.vocab_size = len(keep_words)
        # model = UniLMModel.from_pretrained(
        #     args.output_dir,
        #     config=config,
        #     cache_dir=None,
        # )
        # model.to(args.device)

        s1 = '四海网讯，近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人<Paragraph>接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'
        s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

        for s in [s1, s2]:
            print('生成摘要:', gen_sent(s,tokenizer,model,args))
        print()

if __name__ == '__main__':
    main()