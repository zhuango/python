from ctypes import cdll
Serializer = cdll.LoadLibrary('./libSerializer.so')

vectorsDict = "cn_vectors_50.txt";
wordslist = "label_book_new.txt.extract";
    

dictPath = "test_book_new.txt.extract_50.lstmDict";
serializationPath = "test_book_new.txt.extract_50.serialization";

Serializer.test("sdfsf");
Serializer.Serializer(vectorsDict, wordslist,50, dictPath, serializationPath);
