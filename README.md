# Laporan Proyek Machine Learning - Samuel A. Bhasarie

## Domain Proyek

Industri penerbangan merupakan salah satu sektor vital dalam konektivitas global, dengan harga tiket menjadi faktor penentu utama bagi konsumen dalam merencanakan perjalanan. Harga tiket pesawat bersifat sangat dinamis dan dipengaruhi oleh berbagai faktor kompleks seperti waktu pemesanan, musim, maskapai, rute penerbangan, dan lain-lain. Kemampuan untuk memprediksi harga tiket secara akurat dapat memberikan manfaat signifikan, baik bagi calon penumpang yang ingin mendapatkan harga optimal maupun bagi pihak maskapai dalam menentukan strategi penetapan harga yang kompetitif. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga tiket penerbangan di India berdasarkan serangkaian fitur yang tersedia dalam dataset historis.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, proyek ini mencoba menjawab permasalahan berikut:

1. Bagaimana membangun model prediktif yang mampu mengestimasi harga tiket penerbangan berdasarkan berbagai atribut perjalanan seperti maskapai, tanggal perjalanan, rute, durasi, dan jumlah pemberhentian?
2. Fitur-fitur manakah yang paling dominan dalam menentukan fluktuasi harga tiket penerbangan sehingga dapat memberikan wawasan lebih bagi pengguna atau pihak terkait?

### Goals
Tujuan dari proyek ini adalah:

1. Menghasilkan sebuah model machine learning dengan algoritma regresi yang dapat memprediksi harga tiket penerbangan seakurat mungkin, diukur dengan metrik evaluasi seperti R2 Score, Mean Absolute Error (MAE), dan Root Mean Squared Error (RMSE).
2. Mengidentifikasi dan menyajikan fitur-fitur kunci yang paling berpengaruh terhadap penetapan harga tiket pesawat berdasarkan hasil analisis model.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements
Solusi yang diusulkan untuk mencapai tujuan tersebut adalah dengan menerapkan pendekatan machine learning berbasis regresi. Tahapan utama meliputi:

1. Pemahaman Data (Data Understanding): Melakukan analisis eksploratif (EDA) untuk memahami karakteristik dan pola dalam dataset harga tiket penerbangan.
2. Persiapan Data (Data Preparation): Melakukan pembersihan data, rekayasa fitur (misalnya, ekstraksi informasi dari tanggal dan durasi), serta encoding fitur kategorikal agar siap digunakan oleh model.
3. Pemodelan (Modeling): Membangun, melatih, dan melakukan tuning hyperparameter pada beberapa algoritma regresi seperti Linear Regression, Random Forest Regressor, dan XGBoost Regressor.
4. Evaluasi (Evaluation): Mengevaluasi performa model-model tersebut menggunakan metrik R2 Score, MAE, dan RMSE untuk memilih model terbaik. Analisis feature importance juga akan dilakukan untuk mengidentifikasi faktor penentu harga.

## Data Understanding
- Sumber Data: Dataset yang digunakan adalah "[Flight Price Prediction](https://www.kaggle.com/datasets/muhammadbinimran/flight-price-prediction/)" yang diperoleh dari platform Kaggle. Dataset ini umumnya digunakan untuk melatih model prediksi harga tiket penerbangan di India.

- Pembagian Data: Dataset terdiri dari dua file utama:

    Train_set.csv: Digunakan untuk melatih model prediktif. Berisi fitur-fitur penerbangan beserta harga tiketnya.
    Test_set.csv: Digunakan untuk menguji kemampuan prediksi model pada data baru. Berisi fitur-fitur penerbangan yang sama dengan set pelatihan, namun tanpa kolom harga.
- Tujuan: Variabel target yang akan diprediksi adalah Price (harga tiket).

- Dimensi Data:

    - Train_set.csv awalnya memiliki 10683 baris dan 11 kolom baris.
    - Test_set.csv memiliki 2671 baris dan 10 kolom baris.

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Flight Price Prediction dataset adalah sebagai berikut:
Berikut adalah penjelasan untuk setiap kolom (fitur) yang ada dalam dataset Train_set.csv:

- Airline: Nama maskapai penerbangan (Tipe: Kategorikal). Contoh: IndiGo, Jet Airways.
- Date_of_Journey: Tanggal keberangkatan penerbangan (Tipe: Tanggal/String, akan diubah). Contoh: 24/03/2019.
- Source: Kota asal penerbangan (Tipe: Kategorikal). Contoh: Banglore, Kolkata.
- Destination: Kota tujuan penerbangan (Tipe: Kategorikal). Contoh: New Delhi, Cochin.
- Route: Rute penerbangan yang dilalui, menunjukkan kota-kota transit (Tipe: Kategorikal/String). Contoh: BLR → DEL.
- Dep_Time: Waktu keberangkatan (Tipe: Waktu/String, akan diubah). Contoh: 22:20.
- Arrival_Time: Waktu kedatangan (Tipe: Waktu/String, akan diubah). Contoh: 01:10 22 Mar.
- Duration: Durasi total penerbangan (Tipe: String, akan diubah). Contoh: 2h 50m.
- Total_Stops: Jumlah pemberhentian/transit antara kota asal dan tujuan (Tipe: Kategorikal/String, akan diubah).Contoh: non-stop, 1 stop.
- Additional_Info: Informasi tambahan mengenai penerbangan (Tipe: Kategorikal). Contoh: No info, In-flight meal not included.
- Price: Harga tiket penerbangan dalam mata uang Rupee India (INR) (Tipe: Numerik). Ini adalah variabel target kita.

**Exploratory Data Analysis (EDA)**:
Dari proses EDA, beberapa poin terkait kualitas data yang teridentifikasi adalah:

- Nilai Hilang: Terdapat sejumlah kecil nilai hilang (masing-masing 1) pada kolom Route dan Total_Stops di Train_set.csv.
- Tipe Data: Beberapa kolom penting seperti tanggal (Date_of_Journey), waktu (Dep_Time, Arrival_Time), dan durasi (Duration) masih dalam format string dan memerlukan konversi serta parsing ke format yang sesuai.
- Representasi Fitur: Fitur Total_Stops juga perlu diubah dari format string (misalnya, "1 stop") ke format numerik.
- Potensi Redundansi/Kardinalitas Tinggi: Kolom Route memiliki banyak nilai unik dan informasinya sebagian mungkin sudah tercakup dalam Source, Destination, dan Total_Stops. Kolom Additional_Info didominasi oleh satu nilai.

## Data Preparation
Tahap persiapan data bertujuan untuk membersihkan, mentransformasi, dan merekayasa fitur dari dataset mentah sehingga menjadi format yang optimal untuk proses pemodelan machine learning. Proses ini krusial untuk memastikan kualitas input model dan pada akhirnya meningkatkan performa prediksi.

1. Penggabungan Dataset (Sementara)
Untuk menjaga konsistensi dalam proses rekayasa fitur dan encoding (khususnya one-hot encoding), dataset pelatihan (df_train tanpa kolom target Price) dan dataset uji (df_test) digabungkan untuk sementara waktu. Sebuah kolom penanda dataset_type ditambahkan untuk mempermudah pemisahan kembali di akhir tahap ini. Kolom Price dari df_train disimpan terlebih dahulu secara terpisah.

2. Penanganan Nilai yang Hilang (Missing Values)
Berdasarkan analisis pada tahap Data Understanding, ditemukan satu nilai yang hilang pada kolom Route dan satu nilai yang hilang pada kolom Total_Stops dalam dataset pelatihan asli (df_train). Mengingat jumlahnya yang sangat sedikit (hanya mempengaruhi satu baris data), strategi yang diambil adalah menghapus baris tersebut dari df_train sebelum proses penggabungan dataset. Keputusan ini diambil untuk menjaga kesederhanaan proses dan karena kehilangan satu baris data dari ribuan data yang ada dinilai tidak akan berdampak signifikan terhadap performa model. Setelah langkah ini, dataset yang digunakan untuk tahap selanjutnya dipastikan tidak memiliki nilai yang hilang.

3. Rekayasa Fitur (Feature Engineering) & Transformasi
Rekayasa fitur dilakukan untuk mengekstrak informasi yang lebih bermakna dari fitur-fitur yang ada atau mengubahnya ke dalam format yang lebih sesuai untuk algoritma machine learning.

- Date_of_Journey:

    - Kolom ini diubah dari tipe data string menjadi objek datetime.
    - Fitur baru diekstrak, yaitu Journey_Day (hari dalam bulan) dan Journey_Month (bulan). Fitur-fitur ini berpotensi menangkap pola terkait waktu, seperti musim liburan atau perbedaan harga berdasarkan hari/bulan tertentu.
    - Kolom Date_of_Journey asli kemudian dihapus.

- Dep_Time (Waktu Keberangkatan) & Arrival_Time (Waktu Kedatangan):

    - Kedua kolom waktu ini diubah menjadi objek datetime.
    - Fitur baru diekstrak untuk masing-masing, yaitu Dep_Hour, Dep_Min, Arrival_Hour, dan Arrival_Min. Informasi jam dan menit keberangkatan/kedatangan dapat mempengaruhi harga tiket (misalnya, penerbangan pagi buta atau larut malam).
    - Kolom Dep_Time dan Arrival_Time asli kemudian dihapus.

- Duration (Durasi Penerbangan):

    - Kolom Duration yang awalnya berupa string (misalnya, "2h 50m") diubah menjadi representasi numerik tunggal, yaitu total durasi dalam menit (Duration_Total_Minutes). Ini memudahkan model untuk memproses informasi durasi secara kuantitatif.
    - Sebuah fungsi kustom digunakan untuk melakukan parsing dan konversi ini.
    - Kolom Duration asli kemudian dihapus.

- Total_Stops (Jumlah Pemberhentian):

    - Nilai string pada kolom ini (misalnya, "non-stop", "1 stop") dipetakan ke nilai numerik yang merepresentasikan jumlah pemberhentian (0, 1, 2, dst.). Ini dilakukan karena ada hubungan ordinal yang jelas antara jumlah pemberhentian dan biasanya juga dengan harga tiket.

- Route:

    - Kolom Route dihapus dari dataset. Keputusan ini didasarkan pada pertimbangan bahwa kolom ini memiliki kardinalitas yang sangat tinggi (banyak nilai unik) dan informasi yang terkandung di dalamnya sebagian besar sudah terwakili oleh kombinasi fitur Source, Destination, dan Total_Stops. Penghapusan ini juga membantu mengurangi dimensionalitas data.

- Additional_Info:

    - Kolom ini memiliki sebagian besar nilai berupa "No info". Meskipun demikian, kolom ini tetap dipertahankan dan akan di-encode bersama fitur kategorikal lainnya. Model akan menentukan apakah variasi informasi yang ada di luar "No info" memiliki nilai prediktif.

4. Encoding Fitur Kategorikal
Fitur-fitur kategorikal nominal (Airline, Source, Destination, Additional_Info) perlu diubah menjadi representasi numerik agar dapat diproses oleh algoritma machine learning. Teknik yang digunakan adalah One-Hot Encoding menggunakan fungsi pd.get_dummies().

- One-Hot Encoding dipilih karena fitur-fitur ini tidak memiliki urutan atau tingkatan intrinsik. Penggunaan teknik lain seperti Label Encoding dapat menyebabkan model salah menginterpretasikan adanya hubungan ordinal antar kategori.
- Parameter drop_first=True digunakan saat melakukan one-hot encoding untuk menghindari masalah multikolinearitas dengan menghapus satu kategori dari setiap fitur sebagai referensi. Ini menghasilkan k-1 kolom dummy untuk k kategori.

5. Pemisahan Kembali Dataset Training dan Testing
Setelah semua proses transformasi dan encoding selesai pada dataset gabungan (df_all_encoded), dataset tersebut dipisahkan kembali menjadi set pelatihan (df_train_processed) dan set pengujian (df_test_processed) berdasarkan kolom penanda dataset_type. Kolom penanda ini kemudian dihapus. Kolom target Price yang telah disimpan sebelumnya ditambahkan kembali ke df_train_processed.

6. Pembagian Data untuk Validasi Internal dan Penskalaan Fitur

    - **Pembagian Data untuk Validasi Internal:**
    Meskipun dataset utama telah terbagi menjadi set pelatihan (`Train_set.csv`) dan set pengujian (`Test_set.csv`), diperlukan adanya set validasi internal untuk mengevaluasi dan melakukan tuning model tanpa menggunakan `Test_set.csv` secara prematur. Oleh karena itu, setelah semua fitur pada data pelatihan selesai dipersiapkan (`X_train` dan `y_train`), data tersebut dibagi lagi menjadi set pelatihan baru (`X_train_split`) dan set validasi (`X_val`) menggunakan fungsi `train_test_split` dari pustaka `sklearn.model_selection`. Proporsi pembagian yang umum digunakan adalah 80% untuk data pelatihan baru dan 20% untuk data validasi, dengan `random_state` tertentu untuk memastikan reproduktifitas hasil pembagian.


7. Finalisasi Data untuk Modeling
Sebagai langkah akhir dari persiapan data:

- Variabel target Price dipisahkan dari fitur-fitur lainnya pada df_train_processed, menghasilkan X_train (fitur) dan y_train (target).
- Dataset df_test_processed menjadi X_test (fitur untuk prediksi akhir).
- Dilakukan pemeriksaan akhir untuk memastikan semua fitur dalam X_train dan X_test sudah dalam format numerik dan tidak ada lagi nilai yang hilang.

## Modeling
Tahap pemodelan adalah inti dari proyek machine learning, di mana model prediktif dibangun dan dioptimalkan. Tujuan utama pada tahap ini adalah mengembangkan model regresi yang akurat untuk memprediksi harga tiket penerbangan.

1. Pemilihan Algoritma

    Beberapa algoritma regresi berbasis pohon dipertimbangkan dan dievaluasi dalam proyek ini. Algoritma-algoritma yang dicoba meliputi:

    - Decision Tree Regressor:
        
        Cara Kerja Singkat: Model ini membangun struktur seperti pohon keputusan untuk memprediksi nilai kontinu. Dimulai dari root node, data secara rekursif dibagi menjadi subset yang lebih kecil berdasarkan tes pada nilai fitur tertentu. Tujuannya adalah untuk membuat node-node anak yang lebih murni (memiliki varians target yang lebih kecil). Ketika pohon sudah terbangun, prediksi untuk data baru dibuat dengan menelusuri pohon dari root hingga mencapai leaf node, di mana nilai prediksi (biasanya rata-rata dari nilai target sampel di leaf tersebut) diberikan.

    - Random Forest Regressor:

       Cara Kerja Singkat: Ini adalah model ensemble yang terdiri dari banyak Decision Tree. Setiap pohon dalam "hutan" dilatih pada sampel acak dari data pelatihan (dengan penggantian, dikenal sebagai bootstrap aggregating atau bagging). Selain itu, pada setiap pembelahan node di dalam pohon, hanya subset acak dari total fitur yang dipertimbangkan. Prediksi akhir untuk regresi adalah rata-rata dari prediksi semua pohon individu. Pendekatan ini membantu mengurangi varians dan overfitting yang sering terjadi pada satu Decision Tree.
    - Gradient Boosting Regressor:

        Cara Kerja Singkat: Model ensemble ini juga menggunakan banyak Decision Tree, tetapi membangunnya secara sekuensial (aditif) dalam sebuah proses yang disebut boosting. Model pertama dilatih pada data, kemudian model kedua dilatih untuk memperbaiki kesalahan (residual) yang dibuat oleh model pertama. Proses ini berlanjut, di mana setiap pohon baru fokus pada kesalahan prediksi dari gabungan pohon-pohon sebelumnya. Prediksi akhir adalah jumlah dari kontribusi semua pohon, seringkali dengan bobot tertentu (learning rate) untuk setiap pohon.

3. Proses Pelatihan dan Evaluasi Awal Model

    Setiap algoritma yang dipilih dilatih menggunakan set pelatihan baru (X_train_split dan y_train_split). Karena model berbasis pohon umumnya tidak sensitif terhadap skala fitur, data fitur asli (tidak di-scale) digunakan untuk pelatihan model-model ini.

    Performa awal setiap model dievaluasi pada set validasi (X_val dan y_val) menggunakan metrik evaluasi regresi utama: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R-squared (R2) Score.

    | Model | Parameter Awal Kunci | MAE | MSE | RMSE | R2 Score |
    |----------|----------|----------|----------| ---------- | ----------|
    | Decision Tree Regressor  | random_state=42, parameter lain menggunakan nilai default.  | 624.880522   | 2.127513e+06   | 1458.599517 |0.901331|
    | Random Forest Regressor  | n_estimators=100, random_state=42, n_jobs=-1, parameter pohon lain menggunakan nilai default.   | 651.463890   | 2.383323e+06   | 1543.801626 |0.889467|
    | Gradient Boosting Regressor  | n_estimators=100, random_state=42, learning_rate=0.1 (default), parameter pohon lain menggunakan nilai default.   | 1223.947246   | 3.349550e+06 | 1830.177522 |0.844655|

    Dari evaluasi awal, model Random Forest Regressor memberikan hasil paling menjanjikan dengan menunjukkan performa yang baik dan dipilih untuk tahap hyperparameter tuning guna optimasi lebih lanjut.


4. Peningkatan Model dengan Hyperparameter Tuning

    Hyperparameter tuning adalah proses untuk mencari kombinasi hyperparameter terbaik untuk sebuah model guna meningkatkan performanya. Untuk model digunakan Random Forest Regressor, yang kemudian dilakukan tuning menggunakan RandomizedSearchCV dari sklearn.model_selection. RandomizedSearchCV dipilih karena efisiensinya dalam menjelajahi ruang parameter yang besar dibandingkan GridSearchCV dengan mencoba sejumlah kombinasi parameter secara acak.

    Parameter-parameter yang di-tune untuk Random Forest Regressor meliputi:
    - n_estimators: Jumlah pohon dalam hutan. [100, 200, 300]

    - max_depth: Kedalaman maksimum setiap pohon. [None, 10, 20, 30]

    - min_samples_split: Jumlah minimum sampel yang dibutuhkan untuk membagi node. [2, 5, 10]

    - min_samples_leaf: Jumlah minimum sampel pada setiap leaf node. [1, 2, 4]

    - max_features: Jumlah fitur yang dipertimbangkan saat mencari split terbaik. ['sqrt', 'log2', None]

5. Pemilihan Model Terbaik (Final)

    Berdasarkan perbandingan performa pada set validasi (X_val, y_val), baik sebelum maupun sesudah hyperparameter tuning, model Random Forest Regressor dengan hyperparameter hasil tuning dipilih sebagai model final untuk proyek ini.

    Pemilihan ini didasarkan pada pencapaian metrik evaluasi yang baik dengan perbandingan seperti berikut:

    | Model |  MAE | MSE | RMSE | R2 Score|
    |----------|----------|----------|----------| ----------|
    | Random Forest (Tuned) | 633.961197 | 2.071080e+06 | 1439.124907 | 0.903948 |
    | Random Forest (Initial) | 624.880522 | 2.127513e+06 | 1458.599517 | 0.901331

    Setelah di Tuning, R2 Score mengaami peningkatan.

    Model ini menunjukkan keseimbangan yang baik antara kemampuan generalisasi dan akurasi prediksi pada data yang belum pernah dilihat sebelumnya (dalam konteks data validasi).

6. Pelatihan Model Final

    Setelah model terbaik beserta konfigurasi hyperparameter-nya ditentukan, model tersebut dilatih kembali menggunakan keseluruhan dataset pelatihan (X_train dan y_train). Langkah ini bertujuan untuk memaksimalkan pembelajaran model dari seluruh data historis yang tersedia sebelum digunakan untuk membuat prediksi pada data uji (X_test) yang sebenarnya.

    Model final yang telah dilatih ini (final_model) kemudian siap untuk tahap evaluasi lebih lanjut dan pembuatan prediksi.

## Evaluation
Tahap evaluasi bertujuan untuk mengukur seberapa baik performa model final yang telah dikembangkan dalam memprediksi harga tiket penerbangan. Evaluasi dilakukan secara kuantitatif menggunakan berbagai metrik regresi dan secara kualitatif melalui analisis faktor penting serta visualisasi.

Metrik-metrik berikut digunakan untuk mengevaluasi performa model regresi dalam proyek ini:

1. Mean Absolute Error (MAE)

    MAE menghitung selisih absolut untuk setiap prediksi, menjumlahkannya, lalu membagi dengan jumlah total data poin. Karena menggunakan nilai absolut, MAE tidak membedakan arah kesalahan (apakah prediksi terlalu tinggi atau terlalu rendah) dan kurang sensitif terhadap outlier dibandingkan MSE/RMSE.

2. Mean Squared Error (MSE)

    MSE menghitung selisih untuk setiap prediksi, mengkuadratkannya, menjumlahkan semua kuadrat selisih, lalu membagi dengan jumlah total data poin.

3. Root Mean Squared Error (RMSE)

    Setelah MSE dihitung, RMSE diperoleh dengan mengambil akar kuadratnya. Ini membuatnya lebih interpretatif dalam skala yang sama dengan target variabel.

4. R-squared (R2) Score

    R2 Score (koefisien determinasi) mengukur proporsi varians dalam variabel dependen (harga tiket) yang dapat diprediksi dari variabel independen (fitur). Nilainya berkisar dari −∞ hingga 1. Nilai yang lebih mendekati 1 menunjukkan model yang lebih baik dalam menjelaskan variasi data.

Performa Model Final pada Set Validasi
Model final yang dipilih untuk proyek ini adalah Random Forest Regressor dengan hyperparameter hasil tuning. Berikut adalah performa model ini pada set validasi (X_val, y_val):

[![image.png](https://i.postimg.cc/qq9Yn2Q8/image.png)](https://postimg.cc/rRNQ20Bm)

Analisis Faktor Penting (Feature Importance)
Untuk memahami faktor-faktor apa saja yang paling signifikan memengaruhi prediksi harga tiket oleh model Random Forest Regressor (Tuned), dilakukan analisis feature importance. Fitur dengan tingkat kepentingan tertinggi adalah yang paling banyak dipertimbangkan oleh model dalam membuat keputusan.

[![image.png](https://i.postimg.cc/rpkDjbzH/image.png)](https://postimg.cc/K10cv0P7)
Fitur `Duration_Total_Minutes` dan `Journey_Day` memiliki kontribusi tertinggi

Untuk melihat sebaran prediksi model dibandingkan dengan nilai aktualnya, dibuat scatter plot antara harga aktual (y_val) dan harga prediksi (y_pred_val_final) pada set validasi.
[![image.png](https://i.postimg.cc/sxNcZyjq/image.png)](https://postimg.cc/dZGdzbTj)

Model final yang telah dilatih pada keseluruhan data pelatihan (X_train, y_train) kemudian digunakan untuk membuat prediksi harga pada data uji (X_test). Karena Test_set.csv tidak memiliki kolom Price (harga aktual), evaluasi metrik kuantitatif seperti MAE, RMSE, atau R2 score tidak dapat dilakukan secara langsung pada set ini. Performa model pada set validasi (X_val) dianggap sebagai estimasi terbaik mengenai bagaimana model akan berkinerja pada data baru yang belum pernah dilihat. Prediksi yang dihasilkan untuk X_test merupakan output akhir dari model ini.

# Kesimpulan

- Model machine learning mampu memprediksi dengan bai dengan R2 Score bernilai 0.97.
- Fitur `Duration_Total_Minutes` dan `Journey_Day` memiliki pengaruh yang tinggi.
- Random Forest Regressor hasil hyperparameter tuning dipilih sebagai model akhir project ini.
