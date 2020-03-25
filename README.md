# UniLM_summarization

**使用中文BERT预训练模型结合微软的UniLM实现中文文本摘要**

<a href="https://github.com/microsoft/unilm" target="_blank">Unified Language Model Pre-training</a>微软提出的预训练模型，融合了四种LM，即MLM、从左至右的LM、从右至左的LM
及seq2seq LM，在多任务上达到 state-of-the-art。BERT模型因为使用的MLM方式，在
文本生成方面是一大短板，UniLM通过灵活应用mask技术，将文本生成中seq2seq模型
完美融入BERT中，补齐了BERT在文本生成方面的短板


本文在崔一鸣教授开源的中文BERT预训练模型基础上，使用UniLM进行微调，实现中文文本摘要任务。代码基于PyTorch实现，数据集链接：
<a href="https://github.com/microsoft/unilm" target="_blank">TODO</a>。
