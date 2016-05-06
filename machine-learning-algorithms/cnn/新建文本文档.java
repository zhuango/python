	public static void matchWord(HashMap<String, String> srcMap, String wordDictPath, String tag,int wordDimension) throws IOException {

		HashMap<String, String> hash = new HashMap<String, String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(wordDictPath), "utf-8"));
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(tag), "utf-8"));
		String[] dicWord;
		String line = "";
		System.out.println("matching word ");

		while ((line = br.readLine()) != null) {
			if (!line.trim().equals("")) {
				dicWord = line.split(" ");
				for (int i = 0; i < dicWord.length; i++) {
					String embeddingLine = null;
					embeddingLine = srcMap.get(dicWord[i]);
					if (embeddingLine == null) {// 没有找到该词，判断是否是特征
						embeddingLine = "";
						int DIMENSION = 0;
						DIMENSION = wordDimension;
						
						for (int j = 0; j < DIMENSION; j++) {
							double value = Math.random() * 2 - 1;
							DecimalFormat df = new DecimalFormat("0.000000");
							embeddingLine = embeddingLine + df.format(value) + " ";
						}
//						System.out.println(dicWord[i]);
					}
					embeddingLine = embeddingLine.trim();
					if(!hash.containsKey(dicWord[i]))
					hash.put(dicWord[i], embeddingLine);
					
				}
			}
		}

		for (Iterator<Entry<String, String>> iter = hash.entrySet().iterator(); iter.hasNext();) {
			Entry<String,String> entry = (Entry<String,String>) iter.next();
			Object jian = entry.getKey();
			Object zhi = entry.getValue();
			bw.write(jian + "|" + zhi + "\n");
		}
		br.close();
		bw.close();

	}