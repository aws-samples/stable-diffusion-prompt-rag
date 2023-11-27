import streamlit as st
from PIL import Image
#from io import BytesIO
import imgrag_lib as glib #reference to local lib script


###############################################################################
############## Part 2: Setting page title and header ##########################

# Setting page title and header
st.set_page_config(page_title="Generative AI Playground", page_icon=":robot_face:")
st.markdown(
    "<h1 style='text-align: center;'>Write Stable Diffusion prompts with Retrieval Augmented Generation</h1>", 
    unsafe_allow_html=True
)

#image1 = Image.open('./img/RAG4PromptImprovement.png')
#st.image(image1)
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Strugging with writing effective prompts for your Stable Diffusion Model?</p>', unsafe_allow_html=True)

#st.write("Strugging with writing effective AI art prompt for your Stable Diffusion Model? ")
st.write("This demo will improve your text-to-image prompts by using Retrieval Augmented Generation (RAG) with **LangChain**, **FAISS** as Vector database, **Claude V2** for text generation, and  **Titan embedding** for text embedding.")


#st.write("The following list won’t indent no matter what I try:")
st.markdown("Step 1: Semantic Search")
st.markdown("- Provide your prompt")
st.markdown("- Semantic Search for the most relevant prompts in a Prompt Database with 1K prompt examples from [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)")
st.markdown("Step 2: Prompt Generation using LLM")
st.markdown("- Select the prompt you like to use from the searh results")
st.markdown("- Generate new prompt based on the selected prompt using [Claude V2](https://aws.amazon.com/bedrock/claude/)")
st.markdown("Step 3: Image Generation using [Stable Diffusion XL](https://aws.amazon.com/marketplace/pp/prodview-pe7wqwehghdtm?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) with the prompt you like! ")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)


if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
    with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
        st.session_state.vector_index = glib.get_index() #retrieve the index through the supporting library and store in the app's session cache


###############################################################################
############## Part 3: Setting the 1st tab for prompt improvement #############

tab1, tab2 = st.tabs([" **Prompt Improvement**", "**Image Generation**"])

with tab1:
    with st.form('Form1'):
        st.subheader("Prompt Improvement") #subhead for this column

        original_prompt = st.text_input('''**:blue[Type your prompt for Stable Diffusion Model:]**''')
        submitted1 = st.form_submit_button("Send")
        
        if submitted1 and len(original_prompt) > 0:
            list_prompts = glib.sementic_search(index=st.session_state.vector_index, original_prompt=original_prompt)
            
            st.markdown('''**:blue[Below are the relevant prompts in DIFFUSIONDB:]**''')
            for i in range(len(list_prompts)):
                st.write(i, ": " + list_prompts[i])
                        
            number_selected = st.number_input('Which prompt you want to use for further improvement? Insert a number',value=0)
            print(list_prompts[number_selected])
            new_prompt = glib.get_rag_response(original_prompt, list_prompts[number_selected])
            st.markdown('''**:blue[Below is the prompt generated from LLM:]**''')
            st.write(new_prompt)

###############################################################################
############## Part 4: Setting the 2nd tab for Image Generation ###############
with tab2:
    with st.form('Form2'):
        st.subheader("Image generation") #subhead for this column        
        prompt_text = st.text_input('''**:blue[Provide your improved prompt for Stable Diffusion Model:]**''', key="text2image")
        submitted2 = st.form_submit_button("Generate")        
    
        if submitted2 and len(prompt_text) > 0:
            st.markdown(f"""
            This will show an image using **stable diffusion XL ** with prompt - *{prompt_text}* entered:
            """)
            # Create a spinner to show the image is being generated
            with st.spinner('Generating image based on prompt'):
                generated_image = glib.get_image_response(prompt_content=prompt_text) 
                st.success('Generated stable diffusion model')

            # Open and display the image on the site
            #st.image(image)
            st.image(generated_image,caption=prompt_text)
            #prompt = None        

###############################################################################
###############################################################################