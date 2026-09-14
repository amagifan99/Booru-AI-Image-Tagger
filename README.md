#  Image booru Tagger
 <img alt="Image" src="https://github.com/user-attachments/assets/851721c4-4335-45b5-b5f2-26156eb3a231" />


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
