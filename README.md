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

4. modelbestand toevoegen
   download de zip file van het FILM-model en pak deze uit in de hoofdmap van het project, zodat er een map film_model/ ontstaat met daarin de map saved_model/ .

5. Start de applicatie

python app.py
