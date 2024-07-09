# Fake News Prediction

Dataset yang digunakan ada di folder datasets 
Data Training : 
- Hoax : HoaxData-DataTraining.xlsx
- Real : SatkerData-DataTraining.xlsx

Untuk proses learningnya menggunakan 1000 datasets dengan porsi 75 % data training dan 25 % data testing

Cara jalankan :
1. Jalankan via cmd : py fakeReal.py -> untuk memproses data training dan membentuk model
2. Jalankan via cmd : py app.py,
   kemudian buka link http://127.0.0.1:5000/
   Klik tombol "Process and Training" terlebih dahulu untuk melihat hasil Training datanya (yang telah dijalankan di langkah 1)
3. Setelah dilakukan training, klik "Back To Home"
4. Kemudian silakan dicoba input Title dan Description berdasarkan data Testing yand ada di excel data testing
   - Hoax : HoaxData-DataTesting.xlsx
   - Real : SatkerData-DataTesting.xlsx
5. Seharusnya hasil yang keluar sesuai Hoax atau Realnya sesuai datatesting excelnya.
