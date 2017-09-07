#'é' and 'e\u0301' are called “canonical equivalents”,
s1 = 'café'
s2 = 'cafe\u0301'
print("s1: " + s1)
print("s2: " + s2)

print('length s1: ' + str(len(s1)))
print('length s2: ' + str(len(s2)))
print(s1 == s2)

from unicodedata import normalize
from unicodedata import name

#NFC (Normalization Form C) composes the code points to produce the shortest equivalent string
print("normalize s1: " + str(len(normalize('NFC', s1))))
print("normalize s2: " + str(len(normalize('NFC', s2))))

#NFD decomposes, expanding composed characters into base characters and separate combining characters. 
print("normalize s1: " + str(len(normalize("NFD", s1))))
print("normalize s2: " + str(len(normalize("NFD", s2))))

print(normalize("NFC", s1) == normalize("NFC", s2))
print(normalize("NFD", s1) == normalize("NFD", s2))

ohm = '\u2126'
print(ohm)
print(name(ohm))
ohm_c = normalize("NFC", ohm)
print(ohm_c)
print(name(ohm_c))
print(ohm_c == ohm)
print(normalize('NFC', ohm) == normalize('NFC', ohm_c))

# The letter K in the acronym for the other two normalization forms — NFKC and NFKD — stands for “compatibility”. 

half = '½'
half_nfkc = normalize('NFKC', half)
print(half_nfkc)

four_squared = '4²'
four_squared_kc = normalize('NFKC', four_squared)
print(four_squared_kc)

micro = 'µ'
micro_kc = normalize("NFKC", micro)
print(micro, micro_kc)
print(ord(micro), ord(micro_kc))
print(name(micro), name(micro_kc))

micro_cf = micro.casefold()
print(name(micro_cf))
print(micro, micro_cf)

eszett = 'ß'
print(name(eszett))
eszett_cf = eszett.casefold()
print(eszett, eszett_cf)

# For any string s containing only latin1 characters, s.casefold() produces the same result as s.lower(), with only two exceptions: the micro sign 'µ' is changed to the Greek lower case mu (which looks the same in most fonts) and the German Eszett or “sharp s” (ß) becomes “ss”.

from unicodedata import normalize
def nfc_equal(str1, str2):
    return normalize('NFC', str1) == normalize('NFC', str2)
def fold_equal(str1, str2):
    return (normalize('NFC', str1).casefold() == normalize("NFC", str2).casefold())
s1 = 'café'
s2 = 'cafe\u0301'
print(s1 == s2)
print(nfc_equal(s1, s2))
print(nfc_equal('A', 'a'))

s3 = 'Straße'
s4 = 'strasse'
print(s3 == s4)
print(nfc_equal(s3, s4))
print(fold_equal(s3, s4))
print(fold_equal(s1, s2))
print(fold_equal('A', 'a'))
