# 🎬 Slow Motion Video Interpolatie met FILM

Dit project maakt gebruik van het **FILM-model** (Frame Interpolation for Large Motion) van Google om video's vloeiend vertraagd af te spelen door automatisch tussenframes te genereren.

Je kunt een video uploaden via een eenvoudige webinterface, kiezen hoeveel keer je de slow motion wilt vertragen (2x, 4x, 8x), en het resultaat downloaden.

## project starten

1. Maak een virtuele omgeving aan (optioneel maar aanbevolen)

python -m venv venv

2. Activeer de virtuele omgeving

    - macOS/Linux:

    venv/bin/activate

    - Windows:

    venv\Scripts\activate

3. Installeer de vereiste dependencies

pip install -r requirements.txt

4. Download het FILM-model

python download_model.py

5. Start de Flask-applicatie

python app.py