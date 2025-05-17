# 🎬 Slow Motion Video Interpolatie met FILM

Dit project maakt gebruik van het **FILM-model** (Frame Interpolation for Large Motion) van Google om video's vloeiend vertraagd af te spelen door automatisch tussenframes te genereren.

Je kunt een video uploaden via een eenvoudige webinterface, kiezen hoeveel keer je de slow motion wilt vertragen (2x, 4x, 8x), en het resultaat downloaden.

## ⚠️ Belangrijk

- **film_model/** moet handmatig toegevoegd worden (niet in GitHub).
- Werkt op **CPU** (langzamer). Werkt enkel op **GPU** als CUDA/cuDNN beschikbaar zijn.

## 🚀 Project starten

**Vereiste:** Python **3.9**

1. **Maak een virtuele omgeving aan** (optioneel maar aanbevolen)

   python -m venv venv

2. **Activeer de virtuele omgeving**

   _macOS/Linux:_

   source venv/bin/activate

   _Windows:_

   venv\Scripts\activate

3. **Installeer de dependencies**

   pip install -r requirements.txt

4. **Modelbestand toevoegen**

   - Download de zipfile van het FILM-model.
   - Extract deze in de hoofdmap van het project, zodat er een map `film_model/` ontstaat met daarin `saved_model/`.

5. **Start de applicatie**

   python app.py

6. Open je browser en ga naar: [http://localhost:5000]
