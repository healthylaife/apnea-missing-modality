# Multimodal Sleep Apnea Detection with Missing or Noisy Modality


Polysomnography (PSG) is a type of sleep study tool that records multimodal physiological signals and is widely used for purposes such as sleep staging and respiratory event detection. Conventional machine learning methods assume that each sleep study is associated with a fixed set of observed modalities and that all modalities are available for each sample. However, noisy and missing modalities are a common issue in real-world clinical settings. In this study, we propose a comprehensive pipeline aiming to compensate for the missing or noisy modalities when performing sleep apnea detection. Unlike other existing studies, our proposed model works with any combination of available modalities. Our experiments show that the proposed model outperforms other state-of-the-art approaches in sleep apnea detection using various subsets of available data and different levels of noise, and maintains its high performance (AUROC$>$0.9) even in the presence of high levels of noise or missingness. This may be especially relevant in clinical settings where the level of noise and missingness is high (such as pediatric settings).


![Alt text]([image link](https://i.ibb.co/3NJCfy3/Missing-Sleep.jpg))
