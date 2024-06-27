import streamlit as st

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words 

def levenshtein_distance(source, target):
    distances = [[0]*(len(target)+1) for _ in range(len(source)+1)]

    for i in range(len(source)+1):
        distances[i][0] = i 
    for i in range(len(target)+1):
        distances[0][i] = i 
    
    for i in range(1,len(source)+1):
        for j in range(1,len(target)+1):
            if source[i-1] == target[j-1]:
                distances[i][j] = distances[i-1][j-1]
            else:
                del_cost = distances[i-1][j] + 1
                ins_cost = distances[i][j-1] + 1
                sub_cost = distances[i-1][j-1] + 1

                distances[i][j] = min(del_cost, ins_cost, sub_cost)
    return distances[-1][-1] 

vocabs = load_vocab("./data/vocab.txt")

def main():
    st.title('Word Correction using Levenshtein Distance')
    word = st.text_input('Enter a word:')

    if st.button('Calculate'):
        distances = dict()
        for vocab in vocabs:
            distances[vocab] = levenshtein_distance(word,vocab)
        
        sorted_distances = dict(sorted(distances.items(), key=lambda item:item[1]))
        correct_word = list(sorted_distances.keys())[0] 
        st.write('Correct word: ', correct_word)

        col1, col2 = st.columns(2)
        col1.write('Vocabulary:')
        col1.write(vocabs)

        col2.write('Distances:')
        col2.write(sorted_distances)

if __name__ == "__main__":
    main()