## This is short description about Essay Grader machine learning model!

#### Given an Essay of a minimum 100 words, the model is able to classify the essay into one of the three categories and accordingly assign a score:

- Average (B-C)     
- Excellent (A+)    
- Bad (D-F)          


![image](https://user-images.githubusercontent.com/93938450/167906737-8052ee60-1aed-47fa-bb19-9e5ec30c6d11.png)



### Below is the list of essay grader API's features on which score is calculated:

1. #### **Lexical Features:**

    a) **Number of sentences:**

       On an average for a good essay with a word limit of around 600, count of sentences is found to be around 35.

![image](https://user-images.githubusercontent.com/93938450/167547056-7994dc93-85b1-4d70-8f87-d3d59c70ffb4.png)



   b) **Unique words density:**
    
       This is calculated as ==> 'Total unique words' / 'Total number of words'

       Usually, half of the total number of words should be unique inorder for the essay to be considered good interms of Readability.

![image](https://user-images.githubusercontent.com/93938450/167548100-2f413d78-7dfd-40d0-a3a1-4f55673b046f.png)

   c) **Syllablesperword:**
        
      This is calculated as ==> 'Total number of Syllables' / 'Total number of words'.

      On an average, total number of Syllables in an essay should be around '1.4 times' the total number of words in the Essay.
      Anything less makes the essay harder to read!

   We make use of [Syllables API](https://pypi.org/project/syllables/)
       
![image](https://user-images.githubusercontent.com/93938450/167567324-7957d174-a550-4d58-8254-3cc7e17065d5.png)
            
   d) **Average Word Length**
   
    This is calculated as ==> 'Sum of characters in all words/Total number of words'.

    On an average, it is seen that most Essays have words with around 4 characters. Anything less can make an essay harder to read.
   
   ![image](https://user-images.githubusercontent.com/93938450/167568994-4c999d2a-80b0-470b-941e-5b92721eb945.png)


2. #### **Grammatical Features:**

a) **Verb Density:**

    We make use of "Spacy's Parts of Speech Tagging" feature to extract this count.
    
    This is calculated as ==> 'Total number of verbs/Total number of words'
    
![image](https://user-images.githubusercontent.com/93938450/167571324-c6faf061-e053-4a6e-9cdb-9327b3e58ccd.png)
    

    
b) **Noun Density:**

    We make use of "Spacy's Parts of Speech Tagging" feature to extract this count.
    
    This is calculated as ==> 'Total number of nouns/Total number of words'
    
![image](https://user-images.githubusercontent.com/93938450/167572487-d3b0fffa-15b1-4536-8245-0fba93ef26f5.png)


c) **Adjective Density:**

    We make use of "Spacy's Parts of Speech Tagging" feature to extract this count.
    
    This is calculated as ==> 'Total number of adjectives/Total number of words'

![image](https://user-images.githubusercontent.com/93938450/167572617-bf5f33a1-0fda-4173-8b98-95f0c0716e00.png)


d) **Adverb Density:**

    We make use of "Spacy's Parts of Speech Tagging" feature to extract this count.
    
    This is calculated as ==> 'Total number of adverbs/Total number of words'
    
![image](https://user-images.githubusercontent.com/93938450/167572722-44f03d11-d481-4a12-9dee-cb2447d75911.png)

e) **Pronoun Density:**

    We make use of "Spacy's Parts of Speech Tagging" feature to extract this count.
    
    This is calculated as ==> 'Total number of pronouns/Total number of words'
    
![image](https://user-images.githubusercontent.com/93938450/167572819-e2d7efe0-dce5-4628-9be4-d2a89aba4008.png)

f) **Punctuation Density:**

    We make use of "Spacy's Parts of Speech Tagging" feature to extract this count.
    
    This is calculated as ==> 'Total number of punctuations/Total number of words'
    
![image](https://user-images.githubusercontent.com/93938450/167572964-b73b5a1c-03ef-466b-a739-41892ab5f019.png)


3. #### **Sentence Type Features:**

a) **ActiveVoiceSentence Density:**

    Count of Active voice sentences should be more for a good essay that is easier to Read!
    
    This is calculated as ==> 'Total number of active voice sentences/Total number of sentences'
    
 ![image](https://user-images.githubusercontent.com/93938450/167603045-4df07c44-ba42-423a-9492-95440051e1e7.png)


b) **ComplexSentencesDensity:**

     This is calculated as ==> 'Total number of complex sentences/Total number of sentences'

![image](https://user-images.githubusercontent.com/93938450/167603840-c451d797-7639-455b-a982-d282c761aab4.png)

c) **CompoundSentencesDensity:**

     This is calculated as ==> 'Total number of compound sentences/Total number of sentences'

![image](https://user-images.githubusercontent.com/93938450/167604123-eeca658b-6c3f-40f4-b2ab-885d6f809ea7.png)

d) **SimpleSentencesDensity:**

     This is calculated as ==> 'Total number of simple sentences/Total number of sentences'

![image](https://user-images.githubusercontent.com/93938450/167604540-eef6ec30-c55b-4f47-8207-5939a9b95e29.png)


4. #### **Reading stats Feature:**

a) **FleschScore:**
    
    The Flesch reading ease test measures the readability of a text. It uses two variables to determine the readability score:

   -  The average length of your sentences (measured by the number of words).
   -  The average number of syllables per word.

![image](https://user-images.githubusercontent.com/93938450/167606498-5b60be36-3ab5-4ddf-a115-b7fbfe55d190.png)

5. #### **Writing style Feature:**

a) **Mtld (The Measure of Textual Lexical Diversity):**
    
    The Measure of Textual Lexical Diversity employs a sequentual analysis of a sample to estimate an LD score. 
    Conceptually , MTLD reflects the average number of words in a row for which a certain TTR is maintained.
    
    On an average following is the distribution of Mtld for a range of essays:
    
 ![image](https://user-images.githubusercontent.com/93938450/167608802-a70da635-5220-4ed6-acc3-515e843bebbe.png)
 
 b) **Tell Density :**
    
    A good essay uses a language that shows, not tells. In other words, you're supposed to be writing a narrative- some
    story or anecdote about yourself that shows the admission people who you are. You are not a laundry list of awards,
    classes,and extracurricular activities: Those can be saved for the rest of the application.
    
    This is calculated as ==> 'Total number of Tell sentences/Total number of sentences' in the essay

    
    On an average following is the distribution of Tell Density for a range of essays:
    
![image](https://user-images.githubusercontent.com/93938450/167612982-f28e3a5a-8a57-4a5d-8996-7f373c2c7c8e.png)

 c) **Named Entities:**

    A named entity is a real-world object, such as a person, location, organization, product, etc., that can be denoted with
    a proper name. It can be abstract or have a physical existence.
    
    This is calculated as ==> 'Total number of named entities/Total number of words' in the essay

 
 d) **Named Entities per sentence:**
    
    A named entity is a real-world object, such as a person, location, organization, product, etc., that can be denoted with
    a proper name. It can be abstract or have a physical existence.
    
    This is calculated as ==> 'Total number of named entities/Total number of sentences' in the essay

 
 e) **AWL Level:**
    
    This measures how many words in your essay are present in the Academic word list.
    
    This is calculated as ==> 'Total number of named AWL words/Total number of words' in the essay

[The Academic Word List (AWL)](https://www.wgtn.ac.nz/lals/resources/academicwordlist/information)
    
We make use of [awlify package](https://pypi.org/project/awlify/) for this.
 
 f) **CEFR Level:**
    
    The CEFR organises language proficiency in six levels, A1 to C2, which can be regrouped into three 
    broad levels: Basic User, Independent User and Proficient User, and that can be further subdivided 
    according to the needs of the local context. 
    
![image](https://user-images.githubusercontent.com/93938450/167612706-1b2fc053-46e2-47a3-ad26-584d5e3b443e.png)
    

g ) **Type-token ratio (TTR):**

    The ratio of the number of unique word tokens (referred to as types) to the total number of word tokens in a text.
    
    'Root TTR and Corrected TTR' takes the logarithm and square root of the text length instead of the direct word count as denominator.
    
    Lexical variation is measured usingthe typetoken ratio.
    
  We make use of [lexical-diversity package](https://pypi.org/project/lexical-diversity/0.0.3/) for this.
  
f) **Lexical chain feature:**

   *Lexical chain building process*: 
   
    - The semantically related words for the nouns in the text, including synonyms, hypernyms, and hyponyms, are extracted from the WordNet (Miller, 1995). 
    
    - Then for each pair of the nouns in the text, we check whether they are semantically related. Finally, lexical chains are built by linking semantically 
    related nouns in text. 

    The length of a chain is the number of entities contained in the chain.


