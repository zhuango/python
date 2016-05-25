def write_multiple_items(file, separator, *args):
    file.write(separator.join(args))
    
def concat(*args, sep = "/"):
    return sep.join(args)
    
print(concat("earth", "mars", "venus"))
print(concat("earth", "mars", "venus", sep = "."))