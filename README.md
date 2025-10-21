<h1>
  <a href="https://hkilang.github.io/TTS/"><img src="../../../TTS/raw/main/public/assets/favicon-256x256.png" width="84" align="left" /></a>
  <div lang="zh-HK">香港圍頭話及客家話文字轉語音</div>
  <div><sub>Hong Kong Waitau & Hakka Text-to-Speech</sub></div>
</h1>

<p>
  <div lang="zh-HK">本儲存庫包含<a href="https://hkilang.github.io/TTS/"><strong>香港圍頭話及客家話文字轉語音</strong></a>朗讀器後端部分之原始碼。當選擇線上推導模式時，朗讀器程式會透過本應用程式界面要求伺服器進行模型推導，以產生音訊並傳回程式。</div>
  <div>This repository contains the source code of the back-end part of the <a href="https://hkilang.github.io/TTS/"><strong>Hong Kong Waitau & Hakka Text-to-Speech</strong></a> reader. In online inference mode, the server receives request from the TTS application via this API, generates audio by model inference and sends it back to the app as the response.</div>
</p>

## Technical Overview

It is deployed as a [PythonAnywhere](https://www.pythonanywhere.com) instance on `Chaak2.pythonanywhere.com` for use in the main app.

The API accepts URLs in the following format:

> `https://Chaak2.pythonanywhere.com/TTS/${language}/${text}?voice=${voice}&speed=${speed}`

where the parameters are:

* `${language}`, which must be one of `waitau` or `hakka`;
* `${text}`, which is the **romanised** text input, separated by spaces (`%20`) or `+`. There are 7 available punctuation marks: `.`, `,`, `!`, `?`, `…`, `'` and `-`. Separators are required both before and after punctuation marks. Percent-encoding is not mandatory except for the punctuation `?` (`%3F`).  
  Currently, only transliterations in HKILANG's own romanisation system is accepted. **Chinese characters are not yet supported** in the API, so you will need to first convert Chinese text into pronunciation using this app's interface.
* `${voice}`, which may be one of `male` and `female` (optional, defaults to `male`); and
* `${speed}`, which may be any number between 0.5 and 2 (optional, defaults to 1).

`${` and `}` indicate a parameter and should not be included as part of the URL.

(If you deploy your own instance, of course you will need to replace the domain of the URL with yours.)

Failure to follow the format will result in responses with non-2XX status codes.

The pre-trained [PyTorch](https://pytorch.org) machine learning models used for inference are separately published on [the release page](../../releases) due to size limitations. They must be **placed inside the [`data`](./data) folder** in a deployment.

The main implementation of the API is contained in [`application.py`](./application.py). [`symbols.py`](./symbols.py) contains the mapping between each phoneme and its index. The remaining files are a dead code eliminated reduction of [Bert-VITS2](../../../../fishaudio/Bert-VITS2).

For more information on the interaction between the main app and the API, **refer to the [_Audio Generation_](../../../TTS#audio-generation) section** in the README of the TTS repo.

## Licensing

Different from other repos related to this project, this repo is licensed under AGPL-3.0 to align with [Bert-VITS2](../../../../fishaudio/Bert-VITS2)'s, which is not compatible with MIT.
