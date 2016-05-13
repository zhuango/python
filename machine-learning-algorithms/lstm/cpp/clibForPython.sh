name=Serializer
g++ -c -fPIC ${name}.cpp -o ${name}.o
g++ -shared -Wl,-soname,lib${name}.so -o lib${name}.so  ${name}.o
