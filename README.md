#  Image booru Tagger
<img alt="Image" src="https://github.com/user-attachments/assets/9d2f68a6-71c4-486c-a2c4-9b993e3551c9" />


# Video Scanner
Performance: The default `BATCH_SIZE` is 16. Increasing it to 32 or 64 may improve scanning speed on systems with sufficient memory. `(inside video_analyzer.py)`
<details>
  <summary>Spoiler</summary>
  
<img alt="Image" src="https://github.com/user-attachments/assets/7aa8333a-d829-4a09-b1e7-13a971ae5376" />  

  </details>
  
#  Install

Run `INSTALL-AND-RUN.bat` to download the required dependencies/model and launch the app.

#  Features
- Image Tagger —  analyzes images and generates tags with confidence scores.
- Video Analyzer — Scans videos. Frames containing configured NSFW tags are flagged and their timestamps are recorded.
- Flagged Clips —  generates short video clips around flagged timestamps for easier review.
- Batch Processing 
- JSON Export — Save analysis results and detected tags as JSON



# Credits 

- [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)
- [TensorFlow](https://www.tensorflow.org/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) 
- [Pillow](https://python-pillow.org/) 
- [NumPy](https://numpy.org/) 
- curl 
