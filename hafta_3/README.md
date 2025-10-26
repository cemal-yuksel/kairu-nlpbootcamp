## 🚦 NLP Vaka Çalışması Yolculuğu

<p align="center">
  <b>Gerçek e-ticaret ve spam veri setleriyle metin sınıflandırma, temsili ve analizine giden profesyonel yolculuk!</b>
</p>

```mermaid
flowchart TD
    style A1 fill:#D6EAF8,stroke:#2980B9,stroke-width:2px
    style B1 fill:#F9E79F,stroke:#B7950B,stroke-width:2px
    style B2 fill:#D5F5E3,stroke:#229954,stroke-width:2px
    style B3 fill:#FADBD8,stroke:#C0392B,stroke-width:2px
    style B4 fill:#E8DAEF,stroke:#8E44AD,stroke-width:2px
    style B5 fill:#D4E6F1,stroke:#2471A3,stroke-width:2px
    style B6 fill:#FDEBD0,stroke:#CA6F1E,stroke-width:2px
    style B7 fill:#F6DDCC,stroke:#CA6F1E,stroke-width:2px
    style B8 fill:#D1F2EB,stroke:#148F77,stroke-width:2px
    style B9 fill:#F9E79F,stroke:#B7950B,stroke-width:2px
    style Z1 fill:#D5DBDB,stroke:#34495E,stroke-width:2px

    A1([Ham Veri])
    B1([Veri Yükleme<br><i>Toplama & Okuma</i>])
    B2([Temizleme<br><i>Karakter, Boşluk, HTML</i>])
    B3([Ön Analiz<br><i>Keşifsel Veri Analizi</i>])
    B4([Temsil<br><i>TF-IDF, Embedding</i>])
    B5([Modelleme<br><i>Makine Öğrenmesi</i>])
    B6([Değerlendirme<br><i>Accuracy, F1, ROC</i>])
    B7([Sonuçların Yorumlanması])
    B8([Gelişmiş NLP<br><i>Transformers, BERT</i>])
    B9([Raporlama & Sunum])
    Z1([İş Problemi Çözümü])

    A1 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> B6
    B6 --> B7
    B7 --> B8
    B8 --> B9
    B9 --> Z1

    B1 -.-> B3
    B3 -.-> B5
    B4 -.-> B8
    B6 -.-> B9
```

---

## 📊 Proje Özeti

Bu klasörde, **e-ticaret ve spam veri setleri** üzerinde metin sınıflandırma ve temsili için modern NLP teknikleri uygulanmaktadır. 
Çalışmalar, gerçek veriyle, endüstri standardı Python kütüphaneleri (pandas, scikit-learn, transformers, vb.) ve güncel makine öğrenmesi yaklaşımlarıyla yapılmıştır.

### Ana Adımlar:
- Veri setlerinin yüklenmesi ve incelenmesi
- Temizleme ve ön analiz
- TF-IDF ve embedding ile metin temsili
- Makine öğrenmesi ile sınıflandırma
- Sonuçların değerlendirilmesi ve yorumlanması
- Gelişmiş NLP: Transformer tabanlı modeller

---

## 🌟 Vaka Çalışması Aşamaları & Flashcardlar

### 1. **Veri Yükleme (Toplama & Okuma)**
- **Amaç:** Ham e-ticaret ve spam verilerini uygun formata getirmek.
- <div style="border:1px solid #2980B9; border-radius:8px; padding:12px; background:#F4F8FB; margin:10px 0;">
  <b>Soru:</b> Veri yükleme neden kritik bir adımdır ve sürecin başarısına nasıl etki eder?<br>
  <b>Cevap:</b> Ham verinin doğru ve eksiksiz toplanması, sonraki tüm işlemlerin sağlıklı ilerlemesi için gereklidir.
  </div>

---

### 2. **Temizleme (Cleaning)**
- **Aşamalar:**  
  - Özel karakter, sayı, HTML etiketi, gereksiz boşluk temizliği
- **Kod:**
  ```python
  import re
  metin = "<p>Ücretsiz kargo! 2025 fırsatları...</p>"
  temiz = re.sub(r'<.*?>', '', metin)  # HTML etiketlerini kaldır
  temiz = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]', '', temiz)  # özel karakterleri kaldır
  temiz = temiz.strip()
  print(temiz)
  # çıktı: Ücretsiz kargo fırsatları
  ```
- <div style="border:1px solid #229954; border-radius:8px; padding:12px; background:#F4FBF4; margin:10px 0;">
  <b>Soru:</b> Temizleme adımı neden gereklidir ve model performansına nasıl katkı sağlar?<br>
  <b>Cevap:</b> Temizleme işlemi, metindeki gereksiz karakterleri ve gürültüyü ortadan kaldırarak verinin daha anlamlı ve işlenebilir hale gelmesini sağlar.
  </div>

---

### 3. **Ön Analiz (Keşifsel Veri Analizi)**
- **Amaç:** Veri setinin genel yapısını ve dağılımını anlamak.
- <div style="border:1px solid #C0392B; border-radius:8px; padding:12px; background:#FDF2F0; margin:10px 0;">
  <b>Soru:</b> Keşifsel veri analizi neden gereklidir?<br>
  <b>Cevap:</b> Veri setindeki dengesizlikleri, eksikleri ve önemli özellikleri tespit ederek doğru modelleme stratejisi belirlenir.
  </div>

---

### 4. **Temsil (TF-IDF, Embedding)**
- **Aşamalar:**  
  - Metni sayısal vektörlere dönüştürme
- **Kod:**
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vektorizor = TfidfVectorizer()
  X = vektorizor.fit_transform([temiz])
  print(X.toarray())
  ```
- <div style="border:1px solid #8E44AD; border-radius:8px; padding:12px; background:#F7F1FA; margin:10px 0;">
  <b>Soru:</b> TF-IDF ve embedding neden gereklidir?<br>
  <b>Cevap:</b> Metni sayısal vektörlere dönüştürmek, makine öğrenmesi algoritmalarının metni işleyebilmesi için gereklidir.
  </div>

---

### 5. **Modelleme (Makine Öğrenmesi)**
- **Aşamalar:**  
  - Sınıflandırma algoritmalarının uygulanması
- **Kod:**
  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()
  model.fit(X, [1])  # örnek etiket
  tahmin = model.predict(X)
  print(tahmin)
  ```
- <div style="border:1px solid #2471A3; border-radius:8px; padding:12px; background:#F4F8FB; margin:10px 0;">
  <b>Soru:</b> Makine öğrenmesi ile metin sınıflandırmanın avantajları nelerdir?<br>
  <b>Cevap:</b> Otomatik olarak metinleri kategorilere ayırmak, büyük veri setlerinde hızlı ve doğru analiz sağlar.
  </div>

---

### 6. **Değerlendirme (Accuracy, F1, ROC)**
- **Aşamalar:**  
  - Modelin başarısını ölçmek
- **Kod:**
  ```python
  from sklearn.metrics import accuracy_score, f1_score
  print(accuracy_score([1], tahmin))
  print(f1_score([1], tahmin))
  ```
- <div style="border:1px solid #CA6F1E; border-radius:8px; padding:12px; background:#FDF6ED; margin:10px 0;">
  <b>Soru:</b> Model değerlendirme neden önemlidir?<br>
  <b>Cevap:</b> Modelin gerçek performansını ölçmek ve iyileştirme alanlarını belirlemek için gereklidir.
  </div>

---

### 7. **Sonuçların Yorumlanması**
- **Amaç:** Model çıktılarının iş problemlerine etkisini analiz etmek.
- <div style="border:1px solid #CA6F1E; border-radius:8px; padding:12px; background:#FDF6ED; margin:10px 0;">
  <b>Soru:</b> Sonuçların yorumlanması neden gereklidir?<br>
  <b>Cevap:</b> Modelin iş hedeflerine uygunluğunu ve gerçek dünyadaki etkisini anlamak için gereklidir.
  </div>

---

### 8. **Gelişmiş NLP (Transformers, BERT)**
- **Amaç:** Derin öğrenme tabanlı metin temsili ve sınıflandırma.
- <div style="border:1px solid #148F77; border-radius:8px; padding:12px; background:#F0FBF7; margin:10px 0;">
  <b>Soru:</b> Transformer tabanlı modellerin avantajları nelerdir?<br>
  <b>Cevap:</b> Karmaşık dil ilişkilerini öğrenerek daha yüksek doğruluk ve esneklik sağlarlar.
  </div>

---

### 9. **Raporlama & Sunum**
- **Amaç:** Sonuçların görselleştirilmesi ve paylaşılması.
- <div style="border:1px solid #34495E; border-radius:8px; padding:12px; background:#F4F6F7; margin:10px 0;">
  <b>Soru:</b> Raporlama neden önemlidir?<br>
  <b>Cevap:</b> Proje çıktılarının anlaşılır ve etkili şekilde sunulmasını sağlar.
  </div>

---

## 📂 Klasör İçeriği

- `01.nlp-text-classification.ipynb` : E-ticaret veri seti ile metin sınıflandırma uygulaması (ana notebook)
- `02-transormers_tabanlı_metin_temsili.ipynb` : Transformer tabanlı metin temsili ve sınıflandırma
- `03-weekly-case-study.ipynb` : Haftalık vaka çalışması (karma uygulama)
- `ecommerceDataset.csv` : E-ticaret metin veri seti
- `spam.csv` : Spam tespit veri seti

---

## 💡 Kaynaklar

- [scikit-learn Documentation](https://scikit-learn.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Türkçe NLP Kaynakları](https://github.com/ahmetax/tr-nlp-tools)

---

> **NLP ile metin analizi, iş problemlerinin çözümünde güçlü bir araçtır. Doğru veri ve doğru tekniklerle başarıya ulaşılır!**
