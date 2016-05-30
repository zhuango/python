using System;
using System.Collections.Generic;
using System.IO;
namespace wordExtraction
{
    public static class SerializerWSR
    {   
        public static int dimension = 200;
        public static String type = "book";

        public static void Main(string[] args){
            String corpusPath_CN = "/home/laboratory/corpus_WSR/cn/";
            String corpusPath_EN = "/home/laboratory/corpus_WSR/en/";
            String vectorTablePath_CN = "/home/laboratory/corpus/cn_vectorTable/";
            String vectorTablePath_EN = "/home/laboratory/corpus/en_vectorTable/";
            String seriPath = "/home/laboratory/corpus_WSR/Serializer/";
            // 抽取词典，词典中的词向量只包括词向量
            String EnEmbedPath = vectorTablePath_EN + "/en_vectors_"+dimension.ToString()+".txt";
            String CnEmbedPath = vectorTablePath_CN + "/cn_vectors_"+dimension.ToString()+".txt";

            String srcTrainEnPath = corpusPath_EN + "label_"+type+"_new.txt";
            String srcTestEnPath = corpusPath_EN + "test_"+type+"_new.txt";
            String srcTrainCnPath = corpusPath_CN + "label_"+type+"_new.txt";
            String srcTestCnPath = corpusPath_CN + "test_"+type+"_new.txt";

            String desTrainEnPath = seriPath + "semantic_train_"+type+"_en_"+dimension.ToString()+".txt";
            String desTestEnPath = seriPath + "semantic_test_"+type+"_en_"+dimension.ToString()+".txt";
            String desTrainCnPath = seriPath + "semantic_train_"+type+"_cn_"+dimension.ToString()+".txt";
            String desTestCnPath = seriPath + "semantic_test_"+type+"_cn_"+dimension.ToString()+".txt";

            String dictPath = seriPath + "semantic_"+type+"_dict_"+dimension.ToString()+".txt";

            Dictionary<String, String> gloveEmbedding = loadEmbeddings(EnEmbedPath);
            Dictionary<String, String> word2VecEmbedding = loadEmbeddings(CnEmbedPath);

            List<String> dict = new List<String>();
            dict = extract(srcTrainEnPath, desTrainEnPath, gloveEmbedding, dict);
            dict = extract(srcTestEnPath, desTestEnPath, gloveEmbedding, dict);
            dict = extract(srcTrainCnPath, desTrainCnPath, word2VecEmbedding, dict);
            dict = extract(srcTestCnPath, desTestCnPath, word2VecEmbedding, dict);

            StreamWriter dictWriter = File.CreateText(dictPath);
            System.Console.WriteLine("Writing...");
            foreach(String embedding in dict){
                dictWriter.WriteLine(embedding);
            }

            dictWriter.Close();
        }

        public static List<String> extract(String srcPath,
            String desPath, Dictionary<String, String> embedding, List<String> dict){
            // extract dict.
            int rowNum = 0;
            System.Console.WriteLine("Extracting...");
            StreamReader reader = File.OpenText(srcPath);
            StreamWriter writer = File.CreateText(desPath);
            String line = "";
            String outputLine = "";
            Boolean first = true;
            while(!reader.EndOfStream){
                //          System.Console.WriteLine(line);
                line = reader.ReadLine();
                if(line.Length == 0){
                    continue;
                }
                if(line.Equals("<p>")||line.Equals("<n>")||line.Contains("< N >")||line.Contains("< P >")){
                    if(first == true){
                        first = false;
                    }else{
                        outputLine = outputLine.Trim();
                        //                  System.Console.WriteLine(outputLine);
                        writer.WriteLine(outputLine);
                        outputLine = "";
                    }
                    continue;
                }
                if(line.StartsWith("[") && line.EndsWith("]")){
                    line = line.Substring(2, line.Length-4).Trim();
                }
                line = line.Replace("  ", " ");
                line = line.Replace("   ", " ");

                String [] array = line.Split(' ');
                for(int i = 0; i < array.Length; i++){
                    String embedLine = "";
                    if(embedding.ContainsKey(array[i])){
                        embedLine = embedding[array[i]];
                    }else{
                        embedLine = "";
                        for(int j = 0; j < dimension; j++){
                            Random a = new Random(DateTime.Now.Second);
                            double value = a.NextDouble()*2-1;
                            embedLine = embedLine + " "+ value.ToString();
                        }
                        embedLine = embedLine.Trim();
                        //                  System.Console.WriteLine(embedLine.split(" ").length);
                    }
                    if(!dict.Contains(embedLine)){
                        dict.Add(embedLine);
                    }
                    outputLine = outputLine + (dict.IndexOf(embedLine)+1).ToString() + " ";

                    rowNum++;
                    String []testArray = embedLine.Split(' ');
                    if(testArray.Length != dimension){
                        System.Console.WriteLine(testArray.Length+","+rowNum);
                    }
                }
            }
            writer.WriteLine(outputLine.Trim());
            writer.Close();
            reader.Close();
            return dict;
        }

        public static Dictionary<String, String> loadEmbeddings(String embedPath){
            // 加载GloVe/Word2Vec词典
            System.Console.WriteLine("Loading...");
            Dictionary<String, String> srcMap = new Dictionary<String, String>();
            StreamReader reader = File.OpenText(embedPath);
            String line = "";
            while((line = reader.ReadLine()) != null){
                String word = line.Substring(0, line.IndexOf(" "));
                String embed = line.Substring(line.IndexOf(" ")+1);
                embed = embed.Trim();
                srcMap.Add(word, embed);
            }
            reader.Close();
            return srcMap;
        }
    }
}
