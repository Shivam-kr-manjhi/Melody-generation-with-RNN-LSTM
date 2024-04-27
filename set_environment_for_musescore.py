from music21 import environment
us = environment.UserSettings()

# for key in sorted(us.keys()):
#     print(key)
# musescore_path = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
# us['musescoreDirectPNGPath'] = musescore_path
# us['musicxmlPath'] = musescore_path
print(us['musicxmlPath'])

