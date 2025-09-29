<div align="center">

# 📘 Hafta 1: NLP Pipeline Mimarisi ve Leksik Analiz

**Kairu NLP Bootcamp | Modül 1**

</div>

> **Misyon:** Bu ilk modül, Doğal Dil İşleme'nin temel yapı taşlarını anlamak, yapısal olmayan metin verisini makine öğrenmesi modelleri için anlamlı ve işlenebilir bir formata dönüştürme pratiği kazanmak ve Part-of-Speech (POS) etiketleme ile ilk dilbilimsel analiz deneyimini gerçekleştirmek üzere tasarlanmıştır.

---

### 🎯 Haftanın Hedefleri

Bu modülün sonunda aşağıdaki yetkinliklerin kazanılması hedeflenmektedir:

-   ✅ **NLP Temellerine Hakim Olma:** Alanın temel kavramlarını, terminolojisini ve uygulama alanlarını anlama.
-   ✅ **Metin Ön İşleme Pipeline'ı Implemente Etme:** `Tokenization`, `stopword` temizliği, `lemmatization` gibi adımları içeren bir veri ön işleme akışı kurabilme.
-   ✅ **POS Etiketlerini Yorumlama:** Part-of-Speech etiketlerinin ne anlama geldiğini bilme ve bir metin üzerindeki dağılımlarından temel çıkarımlar yapabilme.
-   ✅ **Uçtan Uca Analiz Yeteneği:** Verilen bir metni alıp, temizleyip, analiz edip temel bulguları raporlayabilme.

---

### 🛠️ Teknoloji ve Kütüphaneler

Bu hafta kullanılan temel araçlar ve kütüphaneler:

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-00AAEF?style=for-the-badge&logo=nltk&logoColor=white)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.5+-09A3D5?style=for-the-badge&logo=spaCy&logoColor=white)](https://spacy.io/)

</div>

---

### 📂 Haftanın İçerikleri

Modül boyunca kullanılacak tüm materyaller aşağıda listelenmiştir.

| Dosya Türü | Dosya Adı | Açıklama |
| :--- | :--- | :--- |
| 📓 **Jupyter Notebook** | `Case Study I.ipynb` | Veri okuma, ön işleme, tokenizasyon, POS tagging ve görselleştirmeyi içeren uygulamalı vaka analizi. |
| 🧠 **Jupyter Notebook** | `I. Hafta Ders Notları.ipynb` | Haftanın teorik konularını, temel kavramları ve kütüphane kullanımlarını içeren ders notları. |
| 💾 **Veri Seti** | `train.csv` | Vaka analizinde ve pratik uygulamalarda kullanılan örnek metin verilerini içeren CSV dosyası. |
| 📑 **Referans Dokümanı** | `POS.docx` | İngilizce POS etiketleri, Türkçe karşılıkları ve örneklerini içeren, analizleri yorumlamaya yardımcı tablo. |

---

### 🔬 Teknik Gündem ve İşlenen Konular

| Alan | İşlenen Konular | Anahtar Kavramlar ve Teknikler |
| :--- | :--- | :--- |
| **🌐 NLP'ye Giriş** | • NLP'nin Tanımı ve Kapsamı<br/>• Endüstriyel Uygulama Alanları | `Computational Linguistics`, `Structured vs. Unstructured Data`, `Corpus`, `Lexicon` |
| **⚙️ Metin Ön İşleme** | • Tokenization Yöntemleri<br/>• Veri Temizleme Stratejileri<br/>• Kök ve Gövde Bulma | `Word/Sentence Tokenization`, `Stopword Removal`, `Punctuation Handling`, `Lemmatization`, `Stemming` |
| **🏷️ Part-of-Speech (POS) Tagging** | • POS Etiketlerinin Dilbilimsel Rolü<br/>• Etiketleme Modellerinin Çalışma Prensibi | `Lexical Categories`, `Penn Treebank Tagset`, `NN (Noun)`, `VB (Verb)`, `JJ (Adjective)`, `RB (Adverb)` |

---

###  casework Vaka Analizi: Dr. Watson'ın Konuşması

Bu haftaki ana uygulamamız, kurgusal bir senaryo üzerinden ilerlemektedir:

-   **Senaryo:** Dr. Emily Watson'ın **3 Ocak 2023** tarihinde Paris'te düzenlenen Uluslararası Yapay Zeka Konferansı'ndaki konuşmasından bir kesit.
-   **Amaç:** Bu konuşma metni üzerinde yukarıda öğrenilen tüm ön işleme adımlarını uygulamak, metni POS etiketleri ile zenginleştirmek ve metnin kelime türü dağılımı gibi temel istatistiklerini çıkararak görselleştirmek. Bu çalışma `Case Study I.ipynb` not defterinde detaylı olarak işlenmiştir.

---

### ✅ Kazanılan Yetkinlikler ve Somut Çıktılar

-   **Pipeline Çıktısı:** Verilen ham metinden başlayarak, temizlenmiş ve token'larına ayrılmış bir veri yapısı elde edildi.
-   **Analiz Çıktısı:** Metin içerisindeki her bir token, POS etiketiyle (`NN`, `VB`, `JJ` vb.) etiketlendi.
-   **Raporlama:** Metindeki en sık kullanılan kelime türlerinin (örneğin, "konuşmanın %30'u isimlerden oluşmaktadır") bir frekans dağılımı oluşturuldu ve görselleştirildi.
-   **Temel Oluşturma:** Bu hafta öğrenilen tüm kavram ve teknikler, sonraki haftaların karmaşık konuları için sağlam bir zemin hazırladı.

---

> 📌 **Sonraki Adımlar:** Bu modülde atılan temeller, bir sonraki hafta ele alınacak olan **Dağıtımsal Anlambilim (Distributional Semantics)** ve kelimelerin anlamsal olarak temsil edildiği **Word Embeddings** konularına geçiş için kritik bir öneme sahiptir.
