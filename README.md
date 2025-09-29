<div align="center">

# 🚀 Kairu NLP Bootcamp: Kuramdan Uygulamaya Derinlemesine Doğal Dil İşleme

Bu repo, **Kairu NLP Bootcamp** programı süresince geliştirilen algoritmaların, deneysel çalışmaların ve üretim odaklı NLP pipeline'larının teknik bir dökümantasyonudur.

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ecosystem-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![spaCy](https://img.shields.io/badge/spaCy-Pipeline-09A3D5?style=for-the-badge&logo=spaCy&logoColor=white)](https://spacy.io/)
[![Gensim](https://img.shields.io/badge/Gensim-Topic%20Modeling-brightgreen?style=for-the-badge)](https://radimrehurek.com/gensim/)
[![Jupyter](https://img.shields.io/badge/JupyterLab-Environment-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)

</div>

> **Projenin Felsefesi:** Bu çalışma, Doğal Dil İşleme alanındaki teorik temelleri (örn: **Dağıtımsal Anlambilim Hipotezi**) modern mimarilerle (örn: **Transformer**) birleştiren, **tekrarlanabilir (reproducible) ve yansıtıcı bir araştırma-geliştirme günlüğü** olarak tasarlanmıştır. Her bir modül, bir NLP probleminin CRISP-DM metodolojisine benzer adımlarla (veri anlama, hazırlama, modelleme, değerlendirme) ele alınmasını hedefler.

---

## 📚 Modül Detayları ve Teknik Derinlik

Her hafta, belirli bir yetkinlik setini hedefleyen bağımsız bir modül olarak tasarlanmıştır.

| Modül & Odak Alanı | 🧠 Kuramsal Altyapı | 💻 Uygulamalı Yetkinlikler & Projeler | 🗂️ Haftalık Dosyalar & Kaynaklar |
| :--- | :--- | :--- | :--- |
| **Hafta 1:** <br/> *NLP Pipeline Mimarisi ve Leksik Analiz* | 🏗️ **Yapısal Olmayan Veri:** Metnin makine öğrenmesi için işlenmesi.<br/>🧹 **Tokenization & Normalizasyon:** `Stemming` vs. `Lemmatization`.<br/>🇹🇷 **Morfolojik Analiz:** Türkçe'nin eklemeli yapısı.<br/>📊 **N-gram & BoW:** Temel `feature extraction` teknikleri. | ⚙️ **Preprocessing Pipeline:** Ham metni `scikit-learn` uyumlu formata dönüştüren uçtan uca pipeline inşası.<br/>🏷️ **POS Etiketleme:** Metindeki isim, fiil, sıfat yoğunlukları gibi anlamsal metrikleri çıkarma.<br/>📈 **İstatistiksel Analiz:** Temel metin istatistikleri üretme. | - 📓 **`Case Study I.ipynb`**: Veri okuma, ön işleme, POS tagging ve görselleştirmeyi içeren ilk vaka analizi.<br/>- 🧠 **`I. Hafta Ders Notları.ipynb`**: Teorik ders notları ve temel kod örnekleri.<br/>- 💾 **`train.csv`**: Uygulamalarda kullanılan örnek veri seti.<br/>- 📑 **`POS.docx`**: POS etiketleri ve Türkçe karşılıklarını içeren referans dokümanı. |
| **Hafta 2:** <br/> *Dağıtımsal Anlambilim ve Yoğun Vektör Temsilleri* | 🔗 **Dağıtımsal Hipotez:** Kelimelerin anlamını bağlamdan çıkarma.<br/>🗺️ **Vektör Uzay Modelleri:** `Word2Vec` (CBOW/Skip-Gram).<br/>⚙️ **Optimizasyon:** `Negative Sampling` & `Hierarchical Softmax`.<br/>🤝 **GloVe Mimarisi:** `Co-occurrence` matrisine dayalı mantık. | 🎓 **Model Eğitimi:** `Gensim` ile Türkçe korpus üzerinden özel `Word2Vec` modeli eğitme.<br/>🌐 **Transfer Learning:** Önceden eğitilmiş `GloVe` embedding'lerini projeye entegre etme.<br/>📐 **Vektör Aritmetiği:** Anlamsal ilişkileri (`Kral - Erkek + Kadın`) test etme.<br/>🎨 **Görselleştirme:** `t-SNE` ile kelime kümelerini keşfetme. | - 🧠 **`01.Text Representations.ipynb`**: *Teorik Temeller* - BoW ve TF-IDF'in matematiksel altyapısı.<br/>- 💻 **`02-Word2vec and GloVe.ipynb`**: *Derinlemesine Uygulama* - `Gensim` ile model eğitimi ve GloVe analizi.<br/>- 🧪 **`03 - Case 2 Solution.ipynb`**: *Sentez ve Analiz* - Farklı temsil stratejilerini karşılaştıran vaka analizi.<br/>- 💾 **`IMDB Dataset.csv`**: Vaka çalışması için film yorumları.<br/>- 📄 **`hurriyet.txt`**: Özel model eğitimi için Türkçe metin korpusu.<br/>- 🌐 **`glove.6B.100d.txt`**: Endüstri standardı, önceden eğitilmiş GloVe vektörleri. |
| **Hafta 3:** <br/> *Subword Modelleri ve Transformer Mimarisi* | 🧩 **Subword Tokenizasyonu:** `OOV` problemine `BPE` ve `WordPiece` ile çözüm.<br/>✨ **FastText:** Karakter `n-gram`'ları ile morfolojik zenginliği yakalama.<br/>⚡ **Attention Mekanizması:** Transformer'ların ardındaki temel sezgi.<br/>🤖 **Encoder-Decoder Mimarisi:** `Self-Attention` temelli modern yapı. | 🔄 **Bağlamsal Temsiller:** `Hugging Face` ile `BERT` ve `DistilBERT`'ten `contextualized embedding`'ler elde etme.<br/>⚖️ **Karşılaştırmalı Analiz:** `FastText` ve `BERT` embedding'lerinin sınıflandırma performansını karşılaştırarak bağlamsal temsillerin gücünü analiz etme. | - 📓 **`01_subword_models.ipynb`**: FastText ve BPE tokenizasyonunun incelenmesi.<br/>- 🤖 **`02_intro_to_transformers.ipynb`**: BERT'ten embedding çıkarma ve analiz.<br/>- 💾 **`data/e-commerce_reviews.csv`**: Vaka analizi için ürün yorumları veri seti. |
| **Hafta 4:** <br/> *Metin Sınıflandırma ve Model Değerlendirme* | 📈 **Olasılıksal Modeller:** `Naive Bayes` ve `Bayes Teoremi`.<br/>🧠 **Ardışıl Mimariler:** `RNN`, `LSTM` & `GRU` hücre yapıları.<br/>📉 **Gradient Problemleri:** `Vanishing/Exploding Gradient` sorunu.<br/>🎯 **Değerlendirme Metrikleri:** `Precision`, `Recall`, `F1-Score`, `Confusion Matrix` ve `ROC-AUC`. | 📊 **Kıyaslamalı Modelleme:** `Naive Bayes`, `Bi-LSTM` ve `DistilBERT` ile duygu analizi `classifier`'ları geliştirme.<br/>📝 **Performans Raporlama:** Modelleri `F1-Score` ve `Confusion Matrix` üzerinden analiz ederek her birinin avantaj/dezavantajlarını teknik bir rapor halinde sunma. | - 💻 **`01_classification_baselines.ipynb`**: Naive Bayes ve Logistic Regression uygulamaları.<br/>- 🧠 **`02_deep_learning_classifiers.ipynb`**: Bi-LSTM ve DistilBERT ile model eğitimi.<br/>- 💾 **`data/sentiment_train_test.csv`**: Duygu analizi için tren ve test veri setleri. |
| **Hafta 5:** <br/> *Üretken Modeller ve İleri Düzey Uygulamalar* | 💡 **Transfer Learning:** `Fine-tuning` paradigmaları.<br/>❓ **Soru-Cevap (QA):** `Extractive` vs. `Abstractive` yaklaşımlar.<br/>✍️ **Özetleme (Summarization):** Metin özetleme teknikleri.<br/>🔄 **T5 Mimarisi:** Her NLP görevini `Text-to-Text` problemine dönüştürme felsefesi. | 🗣️ **QA Sistemi:** `Hugging Face pipeline` API'si ile `Extractive QA` sistemi kurma.<br/>📜 **Özetleyici Pipeline:** `T5` veya `BART` modelini `fine-tune` ederek uzun metinlerden `abstractive` özetler üreten bir sistem geliştirme.<br/>💬 **Chatbot Prototipi:** Transformer tabanlı temel bir diyalog sistemi tasarlama. | - ❓ **`01_extractive_qa_with_bert.ipynb`**: BERT tabanlı soru-cevap uygulaması.<br/>- ✍️ **`02_summarization_with_t5.ipynb`**: T5 ile metin özetleme pipeline'ı.<br/>- 💬 **`03_chatbot_prototype.ipynb`**: Temel chatbot geliştirme not defteri. |

---

## 📂 Sistematik Repo Yapısı

Projenin tekrarlanabilirliğini ve modülerliğini sağlamak amacıyla aşağıdaki detaylı dizin yapısı benimsenmiştir:

```bash
kairu-nlp-bootcamp/
├── hafta_1/
│   ├── Case Study I.ipynb              # Vaka analizi 1
│   ├── I. Hafta Ders Notları.ipynb     # Teorik notlar
│   ├── train.csv                       # Veri seti
│   └── POS.docx                        # POS etiketleri referansı
│
├── hafta_2/
│   ├── 01.Text Representations.ipynb   # BoW ve TF-IDF
│   ├── 02-Word2vec and GloVe.ipynb     # Word2Vec ve GloVe uygulaması
│   ├── 03 - Case 2 Solution.ipynb      # Vaka analizi 2
│   ├── IMDB Dataset.csv                # Film yorumları verisi
│   ├── hurriyet.txt                    # Türkçe metin korpusu
│   └── glove.6B.100d.txt               # Önceden eğitilmiş GloVe vektörleri
│
├── hafta_3/
│   ├── 01_subword_models.ipynb
│   └── data/
│       └── e-commerce_reviews.csv
│
├── hafta_4/
│   ├── 01_classification_baselines.ipynb
│   ├── 02_deep_learning_classifiers.ipynb
│   └── data/
│       └── sentiment_train_test.csv
│
├── hafta_5/
│   ├── 01_extractive_qa_with_bert.ipynb
│   ├── 02_summarization_with_t5.ipynb
│   └── 03_chatbot_prototype.ipynb
│
├── requirements.txt                    # Gerekli kütüphaneler ve versiyonları
└── README.md                           # Bu ana döküman
```
