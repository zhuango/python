#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
using namespace std;

string getOneLine(string filename, unsigned int linepos)
{
    ifstream f;
    f.open(filename.c_str());
    unsigned int counter = 0;
    string line;
    while(getline(f, line))
    {
        counter ++;
        if(counter == linepos) break;
    }
    if(counter == linepos)
    {
        return line;
    }
    else
    {
        return "";
    }
}

vector<string> *split(string line)
{
    istringstream toLine(line);
    //vector<string> *tokens = new vector<string>(istream_iterator<string>{toLine}, istream_iterator<string>{});
//    copy(
//        istream_iterator<string>(toLine),
//        istream_iterator<string(),
//        back_inserter(tokens)
//    )
    return NULL;
}

vector<string> *split(string line, char delim)
{
    istringstream toLine(line);
    vector<string> *tokens = new vector<string>();
    string item;
    while(getline(toLine, item, delim))
    {
        tokens->push_back(item);
    }
    return tokens;
}

map<string, string> *generateVectorDict(string vectorDictPath)
{
    map<string, string> *vectors = new map<string, string>();
    
    unsigned int linenumber = 0;
    
    ifstream vectorFile;
    vectorFile.open(vectorDictPath.c_str());
    if(vectorFile.is_open())
    {
        string line = "";
        while(getline(vectorFile, line))
        {
            linenumber++;
            if(linenumber % 5000 == 0)
            {
                cout << "generate vector dict: " << linenumber << endl;
            }
            vector<string> *vals = split(line, ' ');
            string vectorStr = "";
            for(unsigned int i = 1; i < (*vals).size(); i++)
            {
                vectorStr += (*vals)[i] + " ";
            }
            vectors->insert(pair<string, string>((*vals)[0], vectorStr));
			delete vals;
        }
            
    }
    
    return vectors;
}

vector<double> RandInOne(unsigned int size)
{
    vector<double> data;
    for(unsigned int i = 0; i < size; i++)
    {
        double a = rand() % 10000 / 10000.0 - 0.5;
        data.push_back(a);
    }
    return data;
}
template<class T>
string str(T number)
{
    stringstream ss;
    ss << number;
    return ss.str();
}

    void generateSeri(string wordlist, map<string, string> &vectors, unsigned int dimension, string dictPath, string serializationPath)
    {
        map<string, int> dictTable;
        int dictIndex = 0;
        ofstream dictFile(dictPath.c_str());
        ofstream serializationFile(serializationPath.c_str());
        ifstream wordslistFile(wordlist.c_str());

        /////////////////////////
        int linenumber = 0;
        int notFoundCount = 0;
        /////////////////////////
        
        string line;
        while(getline(wordslistFile, line))
        {
            string serialicationNumbersStr = "";
            linenumber += 1;
            if(linenumber % 1000 == 0)
            {
                cout << serializationPath + " generate serialization: " << linenumber << endl;
            }
            vector<string> *words = split(line, ' ');
            string wordVector = "";
            for(unsigned int i = 0; i < words->size(); i++)
            {
                string word = (*words)[i];
                if(dictTable.find(word) == dictTable.end())
                {
                    if(vectors.find(word) != vectors.end())
                    {
                        wordVector = vectors[word];
                    }
                    else
                    {
                        notFoundCount += 1;
                        vector<double> data = RandInOne(dimension);
                        for(vector<double>::iterator iter = data.begin(); iter != data.end(); iter++)
                        {
                            wordVector += str(*iter) + " ";
                        }
                    }
                    dictIndex += 1;
                    dictTable.insert(pair<string, int>(word, dictIndex));
                    wordVector[wordVector.size() - 1] = '\0';
                    dictFile << wordVector.c_str() << "\n";
                }
                serialicationNumbersStr += str(dictTable[word]) + " ";
            }
            serialicationNumbersStr[serialicationNumbersStr.size() - 1] = '\0';
            serializationFile << serialicationNumbersStr.c_str() << "\n";
        }
        cout << serializationPath << " Done, there are " << notFoundCount << " words which are not found." << endl;
        dictFile.close();
        serializationFile.close();
    }

extern "C"
{
    void Serializer(char * vectorDictPath,char * wordlist, unsigned int dimension, char * dictPath, char * serializationPath)
    {
        map<string, string> *vectors = generateVectorDict(vectorDictPath);
        generateSeri(wordlist, *vectors, dimension, dictPath, serializationPath);
    }
}

extern "C"
{
    void test(char * s)
    {
        cout << "This is a test funtion: " << s << endl;
    }
}
void Generate(string corpusPath, string language, string clas, unsigned int wordDimension,map<string, string> &vectors)
{
    //test
    string wordslist = corpusPath + language + "/test_"+clas+"_new.txt.extract";

    string dictPath = corpusPath + language + "/test_"+clas+"_new.txt.extract_"+str(wordDimension) + ".lstmDict";
    string serializationPath = corpusPath + language + "/test_"+clas+"_new.txt.extract_"+str(wordDimension)+ ".serialization";
    generateSeri(wordslist, vectors, wordDimension,dictPath,serializationPath);
    
    //train
    wordslist = corpusPath + language + "/label_"+clas+"_new.txt.extract";

    dictPath = corpusPath + language + "/label_"+clas+"_new.txt.extract_"+str(wordDimension) + ".lstmDict";
    serializationPath = corpusPath + language + "/label_"+clas+"_new.txt.extract_"+str(wordDimension)+ ".serialization";
    generateSeri(wordslist, vectors, wordDimension,dictPath,serializationPath);

}
void SingleProcess(unsigned int wordDimension)
{
    string corpusPath = "G:/liuzhuang/corpus/";
    vector<string> classes;
    classes.push_back("book");
    classes.push_back("music");
    classes.push_back("dvd");
    vector<string> languages;
    languages.push_back("en");
    languages.push_back("cn");
    string vectorsDict_en = corpusPath + languages[0]+"_vectorTable/"+languages[0]+"_vectors_"+ str(wordDimension) +".txt";
    string vectorsDict_cn = corpusPath + languages[1]+"_vectorTable/"+languages[1]+"_vectors_"+ str(wordDimension) +".txt";
    map<string, string> *vectorDicts_en = generateVectorDict(vectorsDict_en);
    map<string, string> *vectorDicts_cn = generateVectorDict(vectorsDict_cn);
    
    for(unsigned int i =0; i < classes.size(); ++i)
    {
        string clas = classes[i];
        Generate(corpusPath, "en", clas, wordDimension, *vectorDicts_en);
        Generate(corpusPath, "cn", clas, wordDimension, *vectorDicts_cn);
    }
	delete vectorDicts_en;
	delete vectorDicts_cn;
}

int main()
{
    //wordDimensions = [50, 100];
    SingleProcess(50);
    SingleProcess(100);
    
    return 0;
}
