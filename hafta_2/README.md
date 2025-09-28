# 🚀 Hafta 2 – Metin Temsilleri ve Word Embeddings

Bu klasör, **Kairu NLP Bootcamp** programının ikinci haftasına ait ders notlarını, uygulamalı çalışmaları ve veri setlerini içerir. Amaç, katılımcılara metin verisini makine öğrenmesi modelleri için anlamlı sayısal temsillere dönüştürme yetkinliği kazandırmaktır. Bu hafta, klasik **seyrek (sparse) vektör** modellerinden, kelimelerin anlamsal ilişkilerini yakalayan **yoğun (dense) vektör** yani **Word Embedding** yöntemlerine geçiş yapacağız.

---

## 🎯 Öğrenme Hedefleri
- Seyrek ve yoğun metin temsil yöntemleri arasındaki temel farkları anlamak
- **Bag-of-Words (BoW)** ve **TF-IDF** gibi frekans tabanlı modelleri uygulamak ve yorumlamak
- **Word2Vec (CBOW & Skip-gram)** ve **GloVe** gibi kelime gömme (word embedding) modellerinin ardındaki temel mantığı kavramak
- `gensim` kütüphanesini kullanarak özel bir Türkçe korpus (`hurriyet.txt`) üzerinde **Word2Vec modeli** eğitmek
- Önceden eğitilmiş **GloVe** kelime vektörlerini yüklemek ve anlamsal analizlerde kullanmak
- Metin temsil yöntemlerini, duygu analizi gibi gerçek dünya problemlerine uygulamak (`IMDB Dataset.csv`)

---

## 📂 Klasör Yapısı ve İçerik
- `01.Text Representations.ipynb` → **Teorik Temeller:** BoW ve TF-IDF gibi temel vektörleştirme tekniklerinin matematiksel altyapısını ve pratik implementasyonlarını içeren başlangıç noktası.
- `02-Word2vec and GloVe.ipynb` → **Derinlemesine Uygulama:** Anlamsal temsillerin gücünü keşfedeceğimiz, `gensim` ile özel model eğitimi ve önceden eğitilmiş GloVe vektörlerinin analizi.
- `03 - Case 2 Solution.ipynb` → **Sentez ve Analiz:** Haftanın tüm konularını birleştiren, örnek bir metin verisi üzerinde farklı metin temsil stratejilerinin performansını karşılaştıran kapsamlı vaka analizi.
- `IMDB Dataset.csv` → Vaka çalışmamızın temelini oluşturan, pozitif ve negatif film yorumlarını içeren zengin veri seti.
- `hurriyet.txt` → Türkçe NLP yeteneklerimizi test etmek ve özelleştirilmiş bir model eğitmek için kullanacağımız yerel metin korpusu.
- `glove.6B.100d.txt` → Milyarlarca kelime üzerinde eğitilmiş, transfer öğrenme (transfer learning) için kullanılacak endüstri standardı GloVe kelime vektörleri.

---

## 🔬 Konular
- **Seyrek Temsiller (Sparse Representations)**
  - Bag of Words (BoW)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - N-Gram Modellemesi ile Bağlam Zenginleştirme

- **Yoğun Temsiller (Dense Representations): Word Embeddings**
  - Word2Vec Mimarisi: CBOW ve Skip-gram
  - GloVe: Global Vectors for Word Representation
  - Önceden Eğitilmiş (Pre-trained) Modellerin Kullanımı

- **Uygulamalı Çalışma (Case Study)**
  - `IMDB Dataset.csv` üzerinde duygu analizi (sentiment analysis)
  - Farklı metin temsil yöntemlerinin model performansına etkisinin karşılaştırılması
  - Türkçe bir korpus (`hurriyet.txt`) ile sıfırdan Word2Vec modeli eğitimi

---

## 🛠️ Teknoloji Yığını (Tech Stack)
- **Python 3.11+** & **Jupyter Notebook**
- **Veri Manipülasyonu:** `pandas`, `numpy`
- **Klasik ML & NLP:** `scikit-learn` (BoW, TF-IDF, Modelleme için)
- **Word Embedding:** `gensim` (Word2Vec eğitimi ve yönetimi için endüstri standardı)
- **Yardımcı Kütüphaneler:** `nltk` (Metin ön işleme için)
- **Veri Görselleştirme:** `matplotlib`, `seaborn` (Sonuçların ve vektör uzaylarının görsel analizi için)

---

## 📊 Beklenen Çıktılar
- Metin verisini BoW ve TF-IDF matrislerine dönüştürme
- Özel bir korpus ile Word2Vec modeli eğitme ve kelimeler arası anlamsal benzerlikleri keşfetme
- Önceden eğitilmiş GloVe vektörlerini kullanarak kelime analojilerini (ör: kral - erkek + kadın = kraliçe) test etme
- Farklı metin temsil yöntemlerinin duygu analizi performansı üzerindeki etkisini karşılaştıran bir analiz raporu

---

📌 Bu hafta öğrendiğimiz yoğun vektör temsilleri (word embeddings), sonraki haftalarda ele alacağımız **Transformer tabanlı modern NLP modelleri (BERT, GPT)** ve bu modellerin temelini oluşturan **dikkat mekanizmaları (attention mechanism)** için kritik bir altyapı sağlayacaktır.
