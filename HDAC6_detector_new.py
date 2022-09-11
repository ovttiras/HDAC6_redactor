######################
# Import libraries
######################
from matplotlib import cm
from rdkit.Chem.Draw import SimilarityMaps
from numpy import loadtxt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import joblib
from IPython.display import HTML
from stmol import showmol
import py3Dmol
from molvs import standardize_smiles
from padelpy import padeldescriptor
from padelpy import from_sdf
from io import StringIO
import io
from functools import partial
from PIL import Image
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import streamlit.components.v1 as components
# import jdk

######################
# Page Title
######################
st.sidebar.write("<h1 style='text-align: center; color: #FF7F50;'> HDAC1 PREDICTOR v.1.0</h1>", unsafe_allow_html=True)

image = Image.open('app_logo.jpg')
st.sidebar.image(image, use_column_width=True)
st.sidebar.write("<h3 style='text-align: center; color: black;'> A machine learning Web application to assess the potential of histone deacetylase 1 (HDAC1) inhibitors.</h1>", unsafe_allow_html=True)
if st.sidebar.button('Application description'):
    st.sidebar.write('The HDAC1  Predictor application provides an alternative method for assessing the potential of chemicals to be Histone deacetylas 1 (HDAC1) inhibitors.  Compound is classified as active if the predicted IC50 value is  lower than mean IC50 value of the reference drug Vorinostat (11.08 nM)  otherwise compound is  labeled as inactive. This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets generated from scientific articles. The consensus models were developed using open-source chemical descriptors based on ECFP4-like Morgan fingerprints and 2D RDKit descriptors, along with the random forest (RF), gradient boosting (GBM), support vector machines (SVM)  algorithms, using Python 3.7. The models were generated applying the best practices for QSAR model development and validation widely accepted by the community. The applicability domain (AD) of the models was calculated as Dcutoff = ⟨D⟩ + Zs, where «Z» is a similarity threshold parameter defined by a user (0.5 in this study) and «⟨D⟩» and «s» are the average and standard deviation, respectively, of all Euclidian distances in the multidimensional descriptor space between each compound and its nearest neighbors for all compounds in the training set. Batch processing is available through https://github.com/ovttiras/HDAC1-inhibitors.')


with open("manual.pdf", "rb") as file:
    btn=st.sidebar.download_button(
    label="Click to download brief manual",
    data=file,
    file_name="manual of HDAC1  Predictor.pdf",
    mime="application/octet-stream"
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# Select and read  saved model
models_option = st.selectbox('Select QSAR models for prediction', ('GBM_ECFP4', 'RF_Padel'))

threshold = 0.45     
if models_option == 'GBM_ECFP4':
    load_model_GBM = pickle.load(open('FP/HDAC6_GBM_ECFP4.pkl', 'rb'))
    st.header('Select input molecular files')
    
    DRAW = st.checkbox('Draw molecule')
    if DRAW:
        _RELEASE = False    
        if not _RELEASE:
            _component_func = components.declare_component(
                    "chemstreamlit",
                    url="http://localhost:3001"
                    )
            
        else:
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            build_dir = os.path.join(parent_dir, "frontend/build")
            _component_func = components.declare_component("my_component", path=build_dir)
        
        def my_component():
            component_value = _component_func()
            return component_value
        
        res = my_component()
        
        compound_smiles = res
        if st.button('PREDICT'):
                smiles=standardize_smiles(compound_smiles)
                m = Chem.MolFromSmiles(smiles)
                # Calculate molecular descriptors
                f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
                def rdkit_numpy_convert(f_vs):
                    output = []
                    for f in f_vs:
                        arr = np.zeros((1,))
                        DataStructs.ConvertToNumpyArray(f, arr)
                        output.append(arr)
                        return np.asarray(output)


                X = rdkit_numpy_convert(f_vs)

                ######################
                # Pre-built model
                ######################

                # Apply model to make predictions
                prediction_GBM = load_model_GBM.predict(X)
                prediction_GBM = np.array(prediction_GBM)
                prediction_GBM = np.where(prediction_GBM == 1, "Active", "Inactive")


                # Estimination AD
                mol = Chem.MolFromSmiles(smiles)
                mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)

                d = {}
                for m in Chem.SDMolSupplier('datasets/HDAC6_ws.sdf'):
                    if m is not None:
                        mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                        d.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))
                df_ECFP4 = pd.DataFrame.from_dict(d).T
                if df_ECFP4[0].max()>=threshold:
                    cpd_AD_vs = "Inside AD"
                else:
                    cpd_AD_vs = "Outside AD"


                st.header('**Prediction results:**')
                st.write('**HDAC1**: ', prediction_GBM[0])
                st.write('**smiles**: ', smiles)
                st.write('**Applicability domain (AD)**: ', cpd_AD_vs)

                # Generate maps of fragment contribution
                
                def getProba(fp, predictionFunction):
                    return predictionFunction((fp,))[0][1]


                def fpFunction(mol, atomId=-1):
                    fp = SimilarityMaps.GetMorganFingerprint(mol,
                                                            atomId=atomId,
                                                            radius=2,
                                                            nBits=1024)
                    return fp


                fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(
                    mol, fpFunction, lambda x: getProba(x, load_model_GBM.predict_proba), colorMap=cm.PiYG_r)
                st.write('**Predicted fragments contribution:**')
                st.pyplot(fig)
                st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC1 inhibitors). The gray isolines separate positive and negative contributions.')
                
                # 3D structure
                st.header('**3D structure of the studied compound:**')
                def makeblock(smi):
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    mblock = Chem.MolToMolBlock(mol)
                    return mblock

                def render_mol(xyz):
                    xyzview = py3Dmol.view()#(width=400,height=400)
                    xyzview.addModel(xyz,'mol')
                    xyzview.setStyle({'stick':{}})
                    xyzview.setBackgroundColor('black')
                    xyzview.zoomTo()
                    showmol(xyzview,height=500,width=500)
                blk=makeblock(compound_smiles)
                render_mol(blk)
                st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

         
                    
         
            



          
    # Read SMILES input
    SMILES = st.checkbox('SMILES notations (*.smi)')
    if SMILES:
        SMILES_input = ""
        compound_smiles = st.text_area("Enter SMILES", SMILES_input)
        if len(compound_smiles)!=0:
            smiles=standardize_smiles(compound_smiles)
            m = Chem.MolFromSmiles(smiles)
            im = Draw.MolToImage(m)
            st.image(im)
        
        
        if st.button('PREDICT COMPOUND FROM SMILES'):
            # Calculate molecular descriptors
            f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
            def rdkit_numpy_convert(f_vs):
                output = []
                for f in f_vs:
                    arr = np.zeros((1,))
                    DataStructs.ConvertToNumpyArray(f, arr)
                    output.append(arr)
                    return np.asarray(output)


            X = rdkit_numpy_convert(f_vs)

            ######################
            # Pre-built model
            ######################

            # Apply model to make predictions
            prediction_GBM = load_model_GBM.predict(X)
            prediction_GBM = np.array(prediction_GBM)
            prediction_GBM = np.where(prediction_GBM == 1, "Active", "Inactive")


            # Estimination AD
            mol = Chem.MolFromSmiles(smiles)
            mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)

            d = {}
            for m in Chem.SDMolSupplier('datasets/HDAC6_ws.sdf'):
                if m is not None:
                    mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                    d.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))
            df_ECFP4 = pd.DataFrame.from_dict(d).T
            if df_ECFP4[0].max()>=threshold:
                cpd_AD_vs = "Inside AD"
            else:
                cpd_AD_vs = "Outside AD"


            st.header('**Prediction results:**')
            st.write('**HDAC1**: ', prediction_GBM[0])
            st.write('**Applicability domain (AD)**: ', cpd_AD_vs)

            # Generate maps of fragment contribution
            
            def getProba(fp, predictionFunction):
                return predictionFunction((fp,))[0][1]


            def fpFunction(mol, atomId=-1):
                fp = SimilarityMaps.GetMorganFingerprint(mol,
                                                        atomId=atomId,
                                                        radius=2,
                                                        nBits=1024)
                return fp


            fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(
                mol, fpFunction, lambda x: getProba(x, load_model_GBM.predict_proba), colorMap=cm.PiYG_r)
            st.write('**Predicted fragments contribution:**')
            st.pyplot(fig)
            st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC1 inhibitors). The gray isolines separate positive and negative contributions.')
            
            # 3D structure
            st.header('**3D structure of the studied compound:**')
            def makeblock(smi):
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                mblock = Chem.MolToMolBlock(mol)
                return mblock

            def render_mol(xyz):
                xyzview = py3Dmol.view()#(width=400,height=400)
                xyzview.addModel(xyz,'mol')
                xyzview.setStyle({'stick':{}})
                xyzview.setBackgroundColor('black')
                xyzview.zoomTo()
                showmol(xyzview,height=500,width=500)
            blk=makeblock(compound_smiles)
            render_mol(blk)
            st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')
                    

    # Read SDF file 
    LOAD = st.checkbox('MDL multiple SD file (*.sdf)')
    if LOAD:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.header('**1. CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:**')
            supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
            failed_mols = []
            all_mols =[]
            wrong_structure=[]
            wrong_smiles=[]
            bad_index=[]
            for i, m in enumerate(supplier):
                structure = Chem.Mol(m)
                all_mols.append(structure)
                try:
                    Chem.SanitizeMol(structure)
                except:
                    failed_mols.append(m)
                    wrong_smiles.append(Chem.MolToSmiles(m))
                    wrong_structure.append(str(i+1))
                    bad_index.append(i)

           
            st.write('Original data: ', len(all_mols), 'molecules')
            # st.write('Kept data: ', len(moldf), 'molecules')
            st.write('Failed data: ', len(failed_mols), 'molecules')
            if len(failed_mols)!=0:
                number =[]
                for i in range(len(failed_mols)):
                    number.append(str(i+1))
                
                
                bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
                bad_molecules = bad_molecules.set_index('No.')
                st.dataframe(bad_molecules)

            # Standardization SDF file
            all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
            records = []
            for i in range(len(all_mols)):
                record = Chem.MolToSmiles(all_mols[i])
                records.append(record)
            
            moldf_n = []
            for i,record in enumerate(records):
                standard_record = standardize_smiles(record)
                m = Chem.MolFromSmiles(standard_record)
                moldf_n.append(m)
           
            st.write('Kept data: ', len(moldf_n), 'molecules')

             # Calculate molecular descriptors
            def calcfp(mol,funcFPInfo=dict(radius=2,nBits=1024,useFeatures=False,useChirality = False)):
                arr = np.zeros((1,))
                fp = GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr

            moldf=pd.DataFrame(moldf_n)
            moldf['Descriptors'] = moldf[0].apply(calcfp)
            X = np.array(list(moldf['Descriptors'])).astype(int)
            
            moldf.drop(columns='Descriptors', inplace=True)

                
            ######################
            # Pre-built model
            ######################

            # Apply model to make predictions
            prediction_GBM = load_model_GBM.predict(X)
            prediction_GBM = np.array(prediction_GBM)
            prediction_GBM = np.where(prediction_GBM == 1, "Active", "Inactive")
        

            # Estimination AD

            
            d_ECFP4 = {}
            for mol in Chem.SDMolSupplier("datasets/HDAC6_ws.sdf"):
                mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)
                for m in moldf_n:
                    if m is not None:
                        mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                        d_ECFP4.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))

            df_ECFP4 = pd.DataFrame.from_dict(d_ECFP4)
            cpd_AD_vs = np.where(df_ECFP4.max() >= threshold, "Inside AD", "Outside AD")





            # Generate maps of fragment contribution
            def getProba(fp, predictionFunction):
                return predictionFunction((fp,))[0][1]


            def fpFunction(m, atomId=-1):
                fp = SimilarityMaps.GetMorganFingerprint(mol,
                                                        atomId=atomId,
                                                        radius=2,
                                                        nBits=1024)
                return fp
            #Print and download common results

            st.header('**2. RESULTS OF PREDICTION:**')
            if st.button('Show results as table'):                       
                number=[]
                for i in range(len(moldf)):
                    number.append(str(i+1))
                
                for i in range(len(moldf)):
                    a= moldf[0]
                    b=list(a)
                
                smiles=[]
                for i in range(len(b)):
                    m = Chem.MolToSmiles(b[i])
                    smiles.append(m)


                pred_beta = pd.DataFrame({'SMILES': smiles, 'HDAC1 activity': prediction_GBM,'Applicability domain (AD)': cpd_AD_vs, 'No.': number}, index=None)
                predictions = pred_beta.set_index('No.')
                st.dataframe(predictions)           
                def convert_df(df):
                    return df.to_csv().encode('utf-8')  
                csv = convert_df(predictions)

                st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )

            for i in range(len(moldf)):
                a= moldf[0]
                b=list(a)
            # Print results for each molecules
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            def callback():
                st.session_state.button_clicked=True

            if (st.button('Show results and map of fragments contribution for each molecule separately', on_click=callback) or st.session_state.button_clicked):
                st.header('**Prediction results:**')

                items_on_page = st.slider('Select number of compounds on page', 1, 15, 3)
                def paginator(label, items, items_per_page=items_on_page, on_sidebar=False):
                              
                # Figure out where to display the paginator
                    if on_sidebar:
                        location = st.sidebar.empty()
                    else:
                        location = st.empty()

                    # Display a pagination selectbox in the specified location.
                    items = list(items)
                    n_pages = len(items)
                    n_pages = (len(items) - 1) // items_per_page + 1
                    page_format_func = lambda i: "Page " + str(i+1)
                    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

                    # Iterate over the items in the page to let the user display them.
                    min_index = page_number * items_per_page
                    max_index = min_index + items_per_page
                    import itertools
                    return itertools.islice(enumerate(items), min_index, max_index)

                for i, mol in paginator("Select a page", b):
                    smi = Chem.MolToSmiles(b[i])
                    mol=b[i]
                    im = Draw.MolToImage(mol)
                    st.write('**COMPOUNDS NUMBER **' + str(i+1) + '**:**')
                    st.write('**2D structure of compound number **' + str(i+1) + '**:**')
                    st.image(im)
                    # 3D structure
                    st.write('**3D structure of compound number **'+ str(i+1) + '**:**')
                    def makeblock(smi):
                        mol = Chem.MolFromSmiles(smi)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        mblock = Chem.MolToMolBlock(mol)
                        return mblock

                    def render_mol(xyz):
                        xyzview = py3Dmol.view()#(width=400,height=400)
                        xyzview.addModel(xyz,'mol')
                        xyzview.setStyle({'stick':{}})
                        xyzview.setBackgroundColor('black')
                        xyzview.zoomTo()
                        showmol(xyzview,height=500,width=500)
                    blk=makeblock(smi)
                    render_mol(blk)
                    st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

                    predictions = pd.DataFrame({'No. compound': i+1,'SMILES': smi, 'HDAC1 activity': prediction_GBM[i],'Applicability domain (AD)': cpd_AD_vs[i]}, index=[0])
                    
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                <style>
                                tbody th {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                   
                    st.table(predictions)           


                    st.write('**Predicted fragments contribution for compound number **'+ str(i+1) + '**:**')
                    fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, load_model_GBM.predict_proba), colorMap=cm.PiYG_r)
                    st.pyplot(fig)
                    st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC1 inhibitors). The gray isolines separate positive and negative contributions.')
                    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

if models_option == 'RF_Padel':
    st.header('Select input molecular files')

       

    # DRAW = st.checkbox('Draw molecule')
    # if DRAW:
    #     _RELEASE = False    
    #     if not _RELEASE:
    #         _component_func = components.declare_component(
    #                 "chemstreamlit",
    #                 url="http://localhost:3001"
    #                 )
            
    #     else:
    #         parent_dir = os.path.dirname(os.path.abspath(__file__))
    #         build_dir = os.path.join(parent_dir, "frontend/build")
    #         _component_func = components.declare_component("my_component", path=build_dir)
        
    #     def my_component():
    #         component_value = _component_func()
    #         return component_value
        
    #     res = my_component()
        
    #     compound_smiles = res
    #     if st.button('PREDICT'):
    #         smiles=standardize_smiles(compound_smiles)
                
    #             # Calculate molecular descriptors
    #         records_ts = []
    #         records_ts.append(smiles)
    #         df_ts = pd.DataFrame(records_ts, columns=["Smiles"])
    #         file_smiles=df_ts.to_csv
            
    #         # df_ts.to_csv('datasets/molecule_ts.smi', sep=',', index=False, header=False)
    #         load_model_RF = pickle.load(open('Padels/HDAC6_RF_padels.pkl', 'rb'))
    #         import glob
    #         xml_files = glob.glob("fingerprints_xml/*.xml")
    #         FP_list = ['AtomPairs2DCount',
    #     'AtomPairs2D',
    #     'EState',
    #     'CDKextended',
    #     'CDK',
    #     'CDKgraphonly',
    #     'KlekotaRothCount',
    #     'KlekotaRoth',
    #     'MACCS',
    #     'PubChem',
    #     'SubstructureCount',
    #     'Substructure']
    #         fp = dict(zip(FP_list, xml_files))
    #         fingerprint = 'KlekotaRoth'
    #         # fingerprint_output_file = ''.join([fingerprint,'.csv'])
    #         fingerprint_descriptortypes = fp[fingerprint]
            
            
    #         padeldescriptor(mol_dir=file_smiles, 
    #             d_file="Padels/KlekotaRoth.csv",
    #             descriptortypes= fingerprint_descriptortypes,
    #             detectaromaticity=True,
    #             standardizenitro=True,
    #             standardizetautomers=True,
    #             threads=2,
    #             removesalt=True,
    #             log=False,
    #             fingerprints=True)
            
    #         descriptors_kr = pd.read_csv('Padels/KlekotaRoth.csv')
    #         x_ts = descriptors_kr.drop('Name', axis=1)

    #         X = np.asarray(x_ts)
            
    #         ######################
    #         # Pre-built model
    #         ######################

            
    #         # Apply model to make predictions
    #         prediction_RF = load_model_RF.predict(X)
    #         prediction_RF = np.where(prediction_RF == 1, "Active", "Inactive")


    #         # Estimination AD
    #         mol = Chem.MolFromSmiles(smiles)
    #         mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)

    #         d = {}
    #         for m in Chem.SDMolSupplier('datasets/HDAC6_ws.sdf'):
    #             if m is not None:
    #                 mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
    #                 d.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))
    #         df_ECFP4 = pd.DataFrame.from_dict(d).T
    #         if df_ECFP4[0].max()>=threshold:
    #             cpd_AD_vs = "Inside AD"
    #         else:
    #             cpd_AD_vs = "Outside AD"
            
    #         st.header('**Prediction results:**')
    #         st.write('**HDAC6**: ', prediction_RF[0])
    #         st.write('**Applicability domain (AD)**: ', cpd_AD_vs)
    #         # 3D structure
    #         st.header('**3D structure of the studied compound:**')
    #         def makeblock(smi):
    #             mol = Chem.MolFromSmiles(smi)
    #             mol = Chem.AddHs(mol)
    #             AllChem.EmbedMolecule(mol)
    #             mblock = Chem.MolToMolBlock(mol)
    #             return mblock

    #         def render_mol(xyz):
    #             xyzview = py3Dmol.view()#(width=400,height=400)
    #             xyzview.addModel(xyz,'mol')
    #             xyzview.setStyle({'stick':{}})
    #             xyzview.setBackgroundColor('black')
    #             xyzview.zoomTo()
    #             showmol(xyzview,height=500,width=500)
    #         blk=makeblock(compound_smiles)
    #         render_mol(blk)
    #         st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

    
    SMILES = st.checkbox('SMILES notations (*.smi)')
    if SMILES:
        SMILES_input = ""
        compound_smiles = st.text_area("Enter SMILES", SMILES_input)
        if len(compound_smiles)!=0:
            smiles=standardize_smiles(compound_smiles)
            m = Chem.MolFromSmiles(smiles)
            im = Draw.MolToImage(m)
            st.image(im)


        if st.button('PREDICT COMPOUND FROM SMILES'):
            # Calculate molecular descriptors
            records_ts = []
            records_ts.append(smiles)
            df_ts = pd.DataFrame(records_ts, columns=["Smiles"])
            s = StringIO()
            df_ts.to_csv(s, header=False)
            my_csv = s.getvalue()
            load_model_RF = pickle.load(open('Padels/HDAC6_RF_padels.pkl', 'rb'))
            import glob
            xml_files = glob.glob("fingerprints_xml/*.xml")
            FP_list = ['AtomPairs2DCount',
        'AtomPairs2D',
        'EState',
        'CDKextended',
        'CDK',
        'CDKgraphonly',
        'KlekotaRothCount',
        'KlekotaRoth',
        'MACCS',
        'PubChem',
        'SubstructureCount',
        'Substructure']
            fp = dict(zip(FP_list, xml_files))
            fingerprint = 'KlekotaRoth'
            # fingerprint_output_file = ''.join([fingerprint,'.csv'])
            fingerprint_descriptortypes = fp[fingerprint]
            
            
            padeldescriptor(mol_dir=my_csv, 
                d_file="Padels/KlekotaRoth.csv",
                descriptortypes= fingerprint_descriptortypes,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False,
                fingerprints=True)
            
            descriptors_kr = pd.read_csv('Padels/KlekotaRoth.csv')
            x_ts = descriptors_kr.drop('Name', axis=1)

            X = np.asarray(x_ts)
            
            ######################
            # Pre-built model
            ######################

            
            # Apply model to make predictions
            prediction_RF = load_model_RF.predict(X)
            prediction_RF = np.where(prediction_RF == 1, "Active", "Inactive")


            # Estimination AD
            mol = Chem.MolFromSmiles(smiles)
            mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)

            d = {}
            for m in Chem.SDMolSupplier('datasets/HDAC6_ws.sdf'):
                if m is not None:
                    mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                    d.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))
            df_ECFP4 = pd.DataFrame.from_dict(d).T
            if df_ECFP4[0].max()>=threshold:
                cpd_AD_vs = "Inside AD"
            else:
                cpd_AD_vs = "Outside AD"
            
            st.header('**Prediction results:**')
            st.write('**HDAC6**: ', prediction_RF[0])
            st.write('**Applicability domain (AD)**: ', cpd_AD_vs)
            # 3D structure
            st.header('**3D structure of the studied compound:**')
            def makeblock(smi):
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                mblock = Chem.MolToMolBlock(mol)
                return mblock

            def render_mol(xyz):
                xyzview = py3Dmol.view()#(width=400,height=400)
                xyzview.addModel(xyz,'mol')
                xyzview.setStyle({'stick':{}})
                xyzview.setBackgroundColor('black')
                xyzview.zoomTo()
                showmol(xyzview,height=500,width=500)
            blk=makeblock(compound_smiles)
            render_mol(blk)
            st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')
                             

    # Read SDF file 
    LOAD = st.checkbox('MDL multiple SD file (*.sdf)')
    if LOAD:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.header('**1. CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:**')
            supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
            failed_mols = []
            all_mols =[]
            wrong_structure=[]
            wrong_smiles=[]
            bad_index=[]
            for i, m in enumerate(supplier):
                structure = Chem.Mol(m)
                all_mols.append(structure)
                try:
                    Chem.SanitizeMol(structure)
                except:
                    failed_mols.append(m)
                    wrong_smiles.append(Chem.MolToSmiles(m))
                    wrong_structure.append(str(i+1))
                    bad_index.append(i)

           
            st.write('Original data: ', len(all_mols), 'molecules')
            # st.write('Kept data: ', len(moldf), 'molecules')
            st.write('Failed data: ', len(failed_mols), 'molecules')
            if len(failed_mols)!=0:
                number =[]
                for i in range(len(failed_mols)):
                    number.append(str(i+1))
                
                
                bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
                bad_molecules = bad_molecules.set_index('No.')
                st.dataframe(bad_molecules)

            # Standardization SDF file
            all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
            records = []
            for i in range(len(all_mols)):
                record = Chem.MolToSmiles(all_mols[i])
                records.append(record)
            
            moldf = []
            for i,record in enumerate(records):
                standard_record = standardize_smiles(record)
                m = Chem.MolFromSmiles(standard_record)
                moldf.append(m)
           
            st.write('Kept data: ', len(moldf), 'molecules')
    
        
            # Calculate molecular descriptors
            import glob
            xml_files = glob.glob("fingerprints_xml/*.xml")
            FP_list = ['AtomPairs2DCount',
        'AtomPairs2D',
        'EState',
        'CDKextended',
        'CDK',
        'CDKgraphonly',
        'KlekotaRothCount',
        'KlekotaRoth',
        'MACCS',
        'PubChem',
        'SubstructureCount',
        'Substructure']
            
            fp = dict(zip(FP_list, xml_files))
            fingerprint = 'KlekotaRoth'
            # fingerprint_output_file = ''.join([fingerprint,'.csv'])
            fingerprint_descriptortypes = fp[fingerprint]
            df_ts = pd.DataFrame(records, columns=["Smiles"])
            df_ts.to_csv('datasets/molecule_ts.smi', sep=',', index=False, header=False)
            load_model_RF = pickle.load(open('Padels/HDAC6_RF_padels.pkl', 'rb'))
            
            
            
            padeldescriptor(mol_dir='datasets/molecule_ts.smi', 
                d_file="Padels/KlekotaRoth.csv",
                descriptortypes= fingerprint_descriptortypes,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False,
                fingerprints=True)
            
            descriptors_kr = pd.read_csv('Padels/KlekotaRoth.csv')
            x_ts = descriptors_kr.drop('Name', axis=1)

            X = np.asarray(x_ts)
            prediction_RF = load_model_RF.predict(X)
            prediction_RF = np.where(prediction_RF == 1, "Active", "Inactive")


            # Estimination AD
            d_ECFP4 = {}
            for mol in Chem.SDMolSupplier("datasets/HDAC6_ws.sdf"):
                mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)
                for m in moldf:
                    if m is not None:
                        mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                        d_ECFP4.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))

            df_ECFP4 = pd.DataFrame.from_dict(d_ECFP4)
            cpd_AD_vs = np.where(df_ECFP4.max() >= threshold, "Inside AD", "Outside AD")
            
            st.header('**RESULTS OF PREDICTION:**')
            if st.button('Show results as table'):                       
                number=[]
                for i in range(len(moldf)):
                    number.append(str(i+1))

                
                smiles=[]
                for i in range(len(moldf)):
                    m = Chem.MolToSmiles(moldf[i])
                    smiles.append(m)


                pred_beta = pd.DataFrame({'SMILES': smiles, 'HDAC1 activity': prediction_RF,'Applicability domain (AD)': cpd_AD_vs, 'No.': number}, index=None)
                predictions = pred_beta.set_index('No.')
                st.dataframe(predictions)           
                def convert_df(df):
                    return df.to_csv().encode('utf-8')  
                csv = convert_df(predictions)

                st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )


            # Print results for each molecules
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            def callback():
                st.session_state.button_clicked=True

            if (st.button('Show results for each molecule separately', on_click=callback) or st.session_state.button_clicked):
                st.header('**Prediction results:**')

                items_on_page = st.slider('Select number of compounds on page', 1, 15, 3)
                def paginator(label, items, items_per_page=items_on_page, on_sidebar=False):
                              
                # Figure out where to display the paginator
                    if on_sidebar:
                        location = st.sidebar.empty()
                    else:
                        location = st.empty()

                    # Display a pagination selectbox in the specified location.
                    items = list(items)
                    n_pages = len(items)
                    n_pages = (len(items) - 1) // items_per_page + 1
                    page_format_func = lambda i: "Page " + str(i+1)
                    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

                    # Iterate over the items in the page to let the user display them.
                    min_index = page_number * items_per_page
                    max_index = min_index + items_per_page
                    import itertools
                    return itertools.islice(enumerate(items), min_index, max_index)

                for i, mol in paginator("Select a page", moldf):
                    smi = Chem.MolToSmiles(moldf[i])
                    mol=moldf[i]
                    im = Draw.MolToImage(mol)
                    st.write('**COMPOUNDS NUMBER **' + str(i+1) + '**:**')
                    st.write('**2D structure of compound number **' + str(i+1) + '**:**')
                    st.image(im)
                    # 3D structure
                    st.write('**3D structure of compound number **'+ str(i+1) + '**:**')
                    def makeblock(smi):
                        mol = Chem.MolFromSmiles(smi)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        mblock = Chem.MolToMolBlock(mol)
                        return mblock

                    def render_mol(xyz):
                        xyzview = py3Dmol.view()#(width=400,height=400)
                        xyzview.addModel(xyz,'mol')
                        xyzview.setStyle({'stick':{}})
                        xyzview.setBackgroundColor('black')
                        xyzview.zoomTo()
                        showmol(xyzview,height=500,width=500)
                    blk=makeblock(smi)
                    render_mol(blk)
                    st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')
                    predictions = pd.DataFrame({'No. compound': i+1,'SMILES': smi, 'HDAC1 activity': prediction_RF[i],'Applicability domain (AD)': cpd_AD_vs[i]}, index=[0])
                    
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                <style>
                                tbody th {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                   
                    st.table(predictions)           
                    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    
st.text('© Oleg Tinkov, 2022')      
