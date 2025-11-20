# Praktikum_9_kode_opencv

Sebelum menjalankan kode, diharapkan untuk membuat folder baru yang meliputi : 
1. Kode Python dari Github ini
2. Model yang telah dilatih dari colab pada praktikum 9 (Download model yang telah disimpan dari Google Colab (Tersimpan di Google Drive))
3. requirements.txt

## 1. Penyimpanan model dari hasil pelatihan (Google Colab)
#### Command :  
```python
model.save("NAMA_FILE_KALIAN.keras")
```

## 2. Melakukan pembuatan Environment Python versi 3 (3.12.1 atau diatas 3.10.1)
#### Command : 
```bash
python3 -m venv env
```

## 3. Menjalankan environment yang telah dibuat
#### Command (Windows) menggunakan CMD :
```bash
\venv\Scripts\activate
```
#### Command (MacOS):
```bash
source venv/bin/activate
```
## 4. Menginstall library atau modul 
#### Command
```bash
pip install -r requirements.txt
```

## Notes :
### Untuk Tensorflow nya tidak dapat berjalan (Tensorflow 2.2.0)
- Pada Windows yang memiliki Microsoft Visual C++ Redistribute 2022 (Dibawah versi ini), diharapkan untuk menginstall/mengupdate Visual C++.
- https://aka.ms/vc14/vc_redist.arm64.exe (untuk versi ARM64)
- https://aka.ms/vc14/vc_redist.x86.exe (untuk versi X86)
- https://aka.ms/vc14/vc_redist.x64.exe (untuk versi x64)

### Alternatives
- Pada file **requirements.txt** ganti **tensorflow** menjadi **tensorflow==2.10.0**
