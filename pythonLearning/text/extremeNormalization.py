import unicodedata
import string

def shave_marks(txt):
    """Remove all diacritic marks"""
    norm_txt = unicodedata.normalize("NFD", txt)
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", shaved)

url = "http://en.wikipedia.org/wiki/S%C3%A3o_Paulo"
url_extremeNor = shave_marks(url)
print(url_extremeNor)

order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
order_shave = shave_marks(order)
print(order_shave)

Greek = 'Ζέφυρος, Zéfiro'
greek_shave = shave_marks(Greek)
print(greek_shave)
