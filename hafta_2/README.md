<div align="center">

# 🚀 Hafta 2: Metin Temsilleri ve Word Embeddings

**Kairu NLP Bootcamp | Modül 2**

</div>

> **Misyon:** Bu modül, metin verisinin ham karakter dizilerinden, makine öğrenmesi algoritmaları için anlamlı sayısal formatlara dönüştürülme sürecini derinlemesine inceler. Klasik frekans tabanlı **seyrek (sparse) vektör** modellerinden, kelimelerin anlamsal ve sözdizimsel ilişkilerini yüksek boyutlu bir uzayda yakalayan **yoğun (dense) vektör** yani **Word Embedding** paradigmalarına geçiş yapılacaktır.

---

### 🎯 Haftanın Hedefleri

Bu modülün sonunda aşağıdaki yetkinliklerin kazanılması hedeflenmektedir:

-   ✅ **Seyrek ve Yoğun Temsillerin Ayrımını Yapma:** İki yaklaşımın matematiksel temellerini, avantajlarını ve dezavantajlarını anlama.
-   ✅ **Frekans Tabanlı Modelleri Implemente Etme:** **Bag-of-Words (BoW)** ve **TF-IDF** vektörleştiricilerini `scikit-learn` ile uygulayabilme ve sonuçlarını yorumlayabilme.
-   ✅ **Word Embedding Mantığını Kavrama:** **Dağıtımsal Hipotez** temelinde **Word2Vec (CBOW & Skip-gram)** ve **GloVe** gibi modellerin sezgisel ve mimari altyapısını açıklayabilme.
-   ✅ **Özelleştirilmiş Model Eğitme:** `gensim` kütüphanesini kullanarak, verilen bir Türkçe metin korpusu (`hurriyet.txt`) üzerinden sıfırdan bir Word2Vec modeli eğitme.
-   ✅ **Transfer Learning Uygulama:** Önceden eğitilmiş, endüstri standardı **GloVe** kelime vektörlerini projelere dahil etme ve anlamsal analizler için kullanma.
-   ✅ **Uçtan Uca Proje Geliştirme:** Farklı metin temsil yöntemlerini, duygu analizi gibi bir sınıflandırma probleminde karşılaştırmalı olarak test etme.

---

### 🛠️ Teknoloji Yığını

Bu hafta kullanılan temel araçlar ve kütüphaneler:

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gensim](https://img.shields.io/badge/Gensim-4.3+-brightgreen?style=for-the-badge)](https://radimrehurek.com/gensim/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-white?style=for-the-badge&logo=matplotlib&logoColor=black)](https://matplotlib.org/)

</div>

---

### 📂 Haftanın İçerikleri ve Stratejik Rolleri

Modül boyunca kullanılacak tüm materyaller ve veri setleri aşağıda açıklanmıştır.

| Dosya Türü | Dosya Adı | Açıklama ve Stratejik Rolü |
| :--- | :--- | :--- |
| 🧠 **Jupyter Notebook** | `01.Text Representations.ipynb` | **Teorik Temeller:** BoW ve TF-IDF'in matematiksel altyapısını ve pratik implementasyonlarını içeren başlangıç noktası. |
| 💻 **Jupyter Notebook** | `02-Word2vec and GloVe.ipynb` | **Derinlemesine Uygulama:** `gensim` ile özel model eğitimi ve önceden eğitilmiş GloVe vektörlerinin analizi. |
| 🧪 **Jupyter Notebook** | `03 - Case 2 Solution.ipynb` | **Sentez ve Analiz:** Haftanın konularını birleştiren ve farklı temsil stratejilerinin performansını karşılaştıran vaka analizi. |
| 💾 **Veri Seti** | `IMDB Dataset.csv` | Vaka çalışmasının temelini oluşturan, pozitif/negatif film yorumlarını içeren zengin veri seti. |
| 📄 **Korpus** | `hurriyet.txt` | Türkçe NLP yeteneklerini test etmek ve sıfırdan bir model eğitmek için kullanılacak yerel metin korpusu. |
| 🌐 **Pre-trained Model** | `glove.6B.100d.txt` | Milyarlarca kelime üzerinde eğitilmiş, `transfer learning` için kullanılacak endüstri standardı GloVe vektörleri. |

---

### 🔬 Teknik Gündem: Seyrek ve Yoğun Vektör Uzayları

| Alan | İşlenen Modeller ve Teknikler | Temel Kavramlar |
| :--- | :--- | :--- |
| **Seyrek Temsiller** | • Bag of Words (BoW)<br/>• TF-IDF Vektörleştirme<br/>• N-Gram Modellemesi | `Sparsity`, `Term Frequency`, `Inverse Document Frequency`, `Vocabulary`, `Document-Term Matrix` |
| **Yoğun Temsiller** | • Word2Vec Mimarisi<br/>• GloVe Mimarisi<br/>• Önceden Eğitilmiş Modeller | `Dense Vectors`, `Distributional Hypothesis`, `CBOW`, `Skip-gram`, `Co-occurrence Matrix`, `Semantic Similarity` |

---

###  casework Vaka Analizleri

Bu hafta iki temel uygulamalı çalışma gerçekleştirilmiştir:

1.  **Sentiment Analysis (İngilizce):**
    -   **Veri Seti:** `IMDB Dataset.csv`
    -   **Amaç:** BoW, TF-IDF ve GloVe vektörlerini kullanarak bir metnin duygu (pozitif/negatif) durumunu tahmin eden modeller geliştirmek ve bu üç temsil yönteminin model doğruluğuna etkisini karşılaştırmak. Bu çalışma `03 - Case 2 Solution.ipynb` içinde detaylandırılmıştır.

2.  **Custom Word2Vec Model Training (Türkçe):**
    -   **Veri Seti:** `hurriyet.txt`
    -   **Amaç:** Türkçe bir metin korpusu üzerinde sıfırdan bir Word2Vec modeli eğitmek. Eğitilen model ile kelimeler arasındaki anlamsal benzerlikleri (`"ankara"` kelimesine en yakın kelimeler) ve analojileri keşfetmek. Bu çalışma `02-Word2vec and GloVe.ipynb` içinde yer almaktadır.

---

### ✅ Kazanılan Yetkinlikler ve Somut Çıktılar

-   **Vektörleştirme:** Ham metin verisini, makine öğrenmesi modellerinin anlayabileceği BoW ve TF-IDF matrislerine başarılı bir şekilde dönüştürme.
-   **Model Eğitimi:** `gensim` kütüphanesini kullanarak özel bir korpus ile tamamen işlevsel bir Word2Vec modeli eğitme ve kaydetme.
-   **Anlamsal Analiz:** Önceden eğitilmiş GloVe vektörlerini kullanarak kelime analojilerini (`kral - erkek + kadın = kraliçe`) test etme ve kelimeler arası kosinüs benzerliğini hesaplama.
-   **Karşılaştırmalı Rapor:** Farklı metin temsil yöntemlerinin, duygu analizi performansı üzerindeki etkisini (doğruluk, F1 skoru vb.) gösteren bir analiz raporu oluşturma.

---

> 📌 **Sonraki Adımlar:** Bu hafta öğrendiğimiz yoğun vektör temsilleri (word embeddings), kelimelerin sabit anlamlarından ziyade bağlama göre değişen anlamlarını yakalayabilen, bir sonraki modülümüzün konusu olan **Transformer tabanlı modern NLP modelleri (BERT, GPT)** için kritik bir altyapı ve geçiş noktası sağlamaktadır.v
