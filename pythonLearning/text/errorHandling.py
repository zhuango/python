city = 'São Paulo'
code_utf8 = city.encode('utf_8')
print(code_utf8)

code_utf16= city.encode('utf_16')
print(code_utf16)

code_iso  = city.encode('iso8859_1')
print(code_iso)

#code_cp437= city.encode('cp437')
code_cp437 = city.encode('cp437', errors='ignore')
print(code_cp437)

code_cp437 = city.encode('cp437', errors='replace')
print(code_cp437)

code_cp437 = city.encode('cp437', errors='xmlcharrefreplace')
print(code_cp437)

octets = b'Montr\xe9al'
code_cp1252 = octets.decode('cp1252')
print(code_cp1252)

code_iso = octets.decode('iso8859_7')
print(code_iso)

code_ko = octets.decode('koi8_r')
print(code_ko)

code_utf8 = octets.decode('utf_8', 'replace')
print(code_utf8)
