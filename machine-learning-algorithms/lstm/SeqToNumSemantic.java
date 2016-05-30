//	package word_document;

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

public class SeqToNumSemantic {

	public static int dimension = 200;
	public static String type = "book";
	
	public static void main(String[] args) throws IOException{
		String corpusPath_CN = "/home/laboratory/corpus_WSR/cn/";
		String corpusPath_EN = "/home/laboratory/corpus_WSR/en/";
		String vectorTablePath_CN = "/home/laboratory/corpus/cn_vectorTable/";
		String vectorTablePath_EN = "/home/laboratory/corpus/en_vectorTable/";
		String seriPath = "/home/laboratory/corpus_WSR/Serializer/";
		// 抽取词典，词典中的词向量只包括词向量
		String EnEmbedPath = vectorTablePath_EN + "/en_vectors_"+dimension+".txt";
		String CnEmbedPath = vectorTablePath_CN + "/cn_vectors_"+dimension+".txt";
		
		String srcTrainEnPath = corpusPath_EN + "label_"+type+"_new.txt";
		String srcTestEnPath = corpusPath_EN + "test_"+type+"_new.txt";
		String srcTrainCnPath = corpusPath_CN + "label_"+type+"_new.txt";
		String srcTestCnPath = corpusPath_CN + "test_"+type+"_new.txt";
		
		String desTrainEnPath = seriPath + "semantic_train_"+type+"_en_"+dimension+".txt";
		String desTestEnPath = seriPath + "semantic_test_"+type+"_en_"+dimension+".txt";
		String desTrainCnPath = seriPath + "semantic_train_"+type+"_cn_"+dimension+".txt";
		String desTestCnPath = seriPath + "semantic_test_"+type+"_cn_"+dimension+".txt";
		
		String dictPath = seriPath + "semantic_"+type+"_dict_"+Integer.toString(dimension)+".txt";
		
		HashMap<String, String> gloveEmbedding = loadEmbeddings(EnEmbedPath);
		HashMap<String, String> word2VecEmbedding = loadEmbeddings(CnEmbedPath);
		
		ArrayList<String> dict = new ArrayList<String>();
		dict = extract(srcTrainEnPath, desTrainEnPath, gloveEmbedding, dict);
		dict = extract(srcTestEnPath, desTestEnPath, gloveEmbedding, dict);
		dict = extract(srcTrainCnPath, desTrainCnPath, word2VecEmbedding, dict);
		dict = extract(srcTestCnPath, desTestCnPath, word2VecEmbedding, dict);
		
		PrintWriter dictWriter = new PrintWriter(dictPath);
		System.out.println("Writing...");
		for(String embedding:dict){
			dictWriter.println(embedding);
		}
		
		dictWriter.close();
	}

	public static ArrayList<String> extract(String srcPath,
			String desPath, HashMap<String, String> embedding, ArrayList<String> dict) throws IOException{
		// extract dict.
		int rowNum = 0;
		System.out.println("Extracting...");
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(srcPath), "utf-8"));
		PrintWriter writer = new PrintWriter(desPath);
		String line = "";
		String outputLine = "";
		boolean first = true;
		while((line = reader.readLine())!=null){
//			System.out.println(line);
			if(line.length() == 0){
				continue;
			}
			if(line.equals("<p>")||line.equals("<n>")||line.contains("< N >")||line.contains("< P >")){
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

			String [] array = line.split(" ");
			for(int i = 0; i < array.length; i++){
				String embedLine = "";
				if((embedLine = embedding.get(array[i])) != null){
					;
				}else{
					embedLine = "";
					for(int j = 0; j < dimension; j++){
						double value = Math.random()*2-1;
						DecimalFormat df = new DecimalFormat("0.000000");
						embedLine = embedLine + " "+ df.format(value);
					}
					embedLine = embedLine.trim();
//					System.out.println(embedLine.split(" ").length);
				}
				if(!dict.contains(embedLine)){
					dict.add(embedLine);
				}
				outputLine = outputLine + Integer.toString(dict.indexOf(embedLine)+1) + " ";
				
				rowNum++;
				String []testArray = embedLine.split(" ");
				if(testArray.length != dimension){
					System.out.println(testArray.length+","+rowNum);
				}
			}
		}
		writer.println(outputLine.trim());
		writer.close();
		reader.close();
		return dict;
	}

	public static HashMap<String, String> loadEmbeddings(String embedPath) throws IOException{
		// 加载GloVe/Word2Vec词典
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
