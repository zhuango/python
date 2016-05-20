import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;

public class SeqToNumSemanticSentiment {

	public static int dimension = 50;
	public static String type = "book";
	public static int wordNum = 0;
	public static String sentiDim = "50d/";
	public static int num = 1;
	
	public static void main(String[] args) throws IOException{
		//-XX:-UseGCOverheadLimit
		//String corpusPath = "G:/liuzhuang/corpus_newDict_AddMoreNegativeWords/";
		//String vectorTablePath = "G:/liuzhuang/corpus/";
		
		String corpusPath = "/home/laboratory/corpus/";
		String vectorTablePath = "/home/laboratory/corpus/";
		
		String IndexEmbedPath = corpusPath + "data/";
		String corpusPath_en = corpusPath + "en/";
		String corpusPath_cn = corpusPath + "cn/";
		String SeriOutput =corpusPath + "Serializer/";
		// extract dict，vector = word vector + sentence vectro.
		String EnEmbedPath = vectorTablePath + "en_vectorTable/en_vectors_"+Integer.toString(dimension)+".txt";
		String CnEmbedPath = vectorTablePath + "cn_vectorTable/cn_vectors_"+Integer.toString(dimension)+".txt";
		String wordPath = SeriOutput+type+"_wordList_"+Integer.toString(dimension)+".txt";
		String enDictPath = corpusPath +"en/SentiWordList_en.txt";
		String cnDictPath = corpusPath +"cn/SentiWordList_cn.txt";
		

		int tempNum = 0;
		PrintWriter wordWriter = new PrintWriter(wordPath);
		//train_en
		String trainEnSentIndexPath = IndexEmbedPath+sentiDim+type+"/"+type+"_train_index_EN.sent";
		String trainEnSentEmbedPath = IndexEmbedPath+sentiDim+type+"/"+type+"_train_embed_EN.sent";
		//test_en
		String testEnSentIndexPath = IndexEmbedPath+sentiDim+type+"/"+type+"_test_index_EN.sent";
		String testEnSentEmbedPath = IndexEmbedPath+sentiDim+type+"/"+type+"_test_embed_EN.sent";
		//train_cn
		String trainCnSentIndexPath = IndexEmbedPath+sentiDim+type+"/"+type+"_train_index_CN.sent";
		String trainCnSentEmbedPath = IndexEmbedPath+sentiDim+type+"/"+type+"_train_embed_CN.sent";
		//test_cn
		String testCnSentIndexPath = IndexEmbedPath+sentiDim+type+"/"+type+"_test_index_CN.sent";
		String testCnSentEmbedPath = IndexEmbedPath+sentiDim+type+"/"+type+"_test_embed_CN.sent";
		
		String srcTrainEnPath = corpusPath_en + "label_"+type+"_new.txt";
		String srcTestEnPath = corpusPath_en + "test_"+type+"_new.txt";
//		String srcTrainCnPath = "E:/workspace/bi_lingual_preprocess/data/train_cn_0122/label_"+type+"_new.txt";
//		String srcTestCnPath = "E:/workspace/bi_lingual_preprocess/data/test_cn_0122/test_"+type+"_new.txt";
		
		String srcTrainCnPath = corpusPath_cn+"label_"+type+"_new.txt";
		String srcTestCnPath = corpusPath_cn+"test_"+type+"_new.txt";
		
		String desTrainEnPath = SeriOutput +"semantic_sentiment_train_"+type+"_en_"+Integer.toString(dimension)+".txt";
		String desTestEnPath = SeriOutput + "semantic_sentiment_test_"+type+"_en_"+Integer.toString(dimension)+".txt";
		String desTrainCnPath = SeriOutput + "semantic_sentiment_train_"+type+"_cn_"+Integer.toString(dimension)+".txt";
		String desTestCnPath = SeriOutput + "semantic_sentiment_test_"+type+"_cn_"+Integer.toString(dimension)+".txt";
		
		String dictPath = SeriOutput + "semantic_sentiment_"+type+"_dict_"+Integer.toString(dimension)+".txt";
		
		ArrayList<String> enSentiWord = loadSentiList(enDictPath);
		ArrayList<String> cnSentiWord = loadSentiList(cnDictPath);
		
		HashMap<String, String> gloveEmbedding= loadEmbeddings(EnEmbedPath);
		HashMap<String, String> word2VecEmbedding= loadEmbeddings(CnEmbedPath);
		//train_en
		HashMap<String, String> trainEnSentEmbedding = mapSentEmbedding(trainEnSentIndexPath, trainEnSentEmbedPath);
		//test_en
		HashMap<String, String> testEnSentEmbedding = mapSentEmbedding(testEnSentIndexPath, testEnSentEmbedPath);
		//train_cn
		HashMap<String, String> trainCnSentEmbedding = mapSentEmbedding(trainCnSentIndexPath, trainCnSentEmbedPath);
		//test_cn
		HashMap<String, String> testCnSentEmbedding = mapSentEmbedding(testCnSentIndexPath, testCnSentEmbedPath);
		
		ArrayList<String> dict = new ArrayList<String>();
		ArrayList<String> wordIndex = new ArrayList<String>();
		HashMap<String,String> wordList = new HashMap<String, String>();
		
		extract(srcTrainEnPath, desTrainEnPath, gloveEmbedding, trainEnSentEmbedding, dict, wordList, wordIndex, enSentiWord);
		System.out.println(wordNum-tempNum);
		tempNum = wordNum; 
		extract(srcTestEnPath, desTestEnPath, gloveEmbedding, testEnSentEmbedding, dict, wordList, wordIndex,enSentiWord);
		System.out.println(wordNum-tempNum);
		tempNum = wordNum; 
		extract(srcTrainCnPath, desTrainCnPath, word2VecEmbedding, trainCnSentEmbedding, dict, wordList, wordIndex, cnSentiWord);
		System.out.println(wordNum-tempNum);
		tempNum = wordNum; 
		extract(srcTestCnPath, desTestCnPath, word2VecEmbedding, testCnSentEmbedding, dict, wordList, wordIndex, cnSentiWord);
		System.out.println(wordNum-tempNum);
		tempNum = wordNum; 
		
		PrintWriter dictWriter = new PrintWriter(dictPath);
		System.out.println("Writing...");
		for(String embedding:dict){
			dictWriter.println(embedding);
		}
		
		for(String word:wordIndex){
			wordWriter.println(word);
		}
		
		wordWriter.close();
		dictWriter.close();
	}

	public static ArrayList<String> loadSentiList(String dictPath) throws IOException{
		//load sentiment word list.
		BufferedReader reader = new BufferedReader(new FileReader(new File(dictPath)));
		ArrayList<String> wordList = new ArrayList<String>();
		String line = "";
		while((line = reader.readLine())!=null){
			line = line.trim();
			wordList.add(line);
		}
		
		reader.close();
		return wordList;
	}

	public static void extract(String srcPath,
			String desPath, HashMap<String, String> gloveEmbedding,
			HashMap<String, String> sentEmbedding, ArrayList<String> dict, HashMap<String, String> wordList, ArrayList<String> list, ArrayList<String> sentiList) throws IOException{

		int rowNum = 0;
		System.out.println("Extracting...");
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(srcPath), "utf-8"));
		PrintWriter writer = new PrintWriter(desPath);
		String line = "";
		String outputLine = "";
		int docNum = 0;
		int sentNum = 0;
		boolean first = true;
		while((line = reader.readLine())!=null){
//			System.out.println(line);
			if(line.length() == 0){
				continue;
			}
			if(line.equals("<p>")||line.equals("<n>")||line.contains("< N >")||line.contains("< P >")){
				docNum++;
				sentNum = 0;
				if(first == true){
					first = false;
				}else{
					outputLine = outputLine.trim();
//					System.out.println(outputLine);
					writer.println(outputLine);
					outputLine = "";
				}
				continue;
			}
			if(line.startsWith("[") && line.endsWith("]")){
				line = line.substring(2, line.length()-2);
			}
			line = line.replaceAll("  ", " ");
			line = line.replaceAll("   ", " ");
//			if(srcPath.equals("E:/workspace/bi_lingual_preprocess/data/train_cn_0122/label_book_new.txt")){
//				System.out.println(line);
//			}
			String [] array = line.split(" ");
			sentNum++;
			for(int i = 0; i < array.length; i++){
				outputLine = outputLine + Integer.toString(++wordNum) + " ";
				if(wordList.containsKey(array[i])){
					String isSentiWord = "0";
					if(sentiList.contains(array[i])){
						isSentiWord = "1";
					}
					list.add(wordList.get(array[i]) + " " +isSentiWord);
				}else{
					wordList.put(array[i], "***"+Integer.toString(num++));
					String isSentiWord = "0";
					if(sentiList.contains(array[i])){
						isSentiWord = "1";
					}
					list.add(wordList.get(array[i]) + " " +isSentiWord);
				}

				String embedding = "";
				if((embedding = gloveEmbedding.get(array[i]))!=null){
					//get the embedding from gloveEmbeddings
					embedding = embedding + " " + sentEmbedding.get(docNum+" "+sentNum);
				}else{
					//not get the embedding from gloveEmbeddings
					embedding = "";
					for(int j = 0; j < dimension; j++){
						double value = Math.random()*2-1;
						DecimalFormat df = new DecimalFormat("0.000000");
						embedding = embedding + " "+ df.format(value);
					}
					embedding = embedding + " " + sentEmbedding.get(docNum+" "+sentNum);
				}
				embedding = embedding.trim();
				dict.add(embedding);
				rowNum++;
				String []testArray = embedding.split(" ");
//				if(testArray.length != 150){
//					System.out.println(rowNum);
//					System.out.println(docNum+"_"+sentNum);
//				}
			}
		}
		writer.println(outputLine.trim());
		
		writer.close();
		reader.close();
	}

	public static HashMap<String, String> mapSentEmbedding(
			String sentIndexPath, String sentEmbedPath) throws IOException{
		// 将句子情感标识和index匹配
		System.out.println("Mapping...");
		HashMap<String, String> sentMap = new HashMap<String, String>();
		BufferedReader indexReader = new BufferedReader(new FileReader(new File(sentIndexPath)));
		BufferedReader embedReader = new BufferedReader(new FileReader(new File(sentEmbedPath)));
		String indexLine = "";
		String embedLine = "";
		while((indexLine = indexReader.readLine())!=null && (embedLine = embedReader.readLine())!=null){
			embedLine = embedLine.replaceAll("	", " ");
			embedLine = embedLine.trim();
			sentMap.put(indexLine, embedLine);
		}
		indexReader.close();
		embedReader.close();
		return sentMap;
	}

	public static HashMap<String, String> loadEmbeddings(String embedPath) throws IOException{
		// 加载GloVe词典
		System.out.println("Loading...");
		HashMap<String, String> srcMap = new HashMap<String, String>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(embedPath), "utf-8"));
		String line = "";
		while((line = reader.readLine()) != null){
			String word = line.substring(0, line.indexOf(" "));
			String embed = line.substring(line.indexOf(" ")+1, line.length());
			embed = embed.trim();
			srcMap.put(word, embed);
		}
		reader.close();
		return srcMap;
	}

}
