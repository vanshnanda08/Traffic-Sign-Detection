"""P57 local upload application. No training/evaluation/install UI actions."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
import streamlit as st
from PIL import Image
from p57_ui.core import (CLASSES, normalize, png, sha, pixel_sha, regions_valid,
                         state_key, displayed, region_preview, exports, model_manifest)
from p57_ui.client import call_worker

st.set_page_config(page_title='P57 | Traffic signs', page_icon='🚦', layout='wide')
st.markdown('<style>html,body,[class*="css"],h1,h2,h3 {font-family:Arial,Helvetica,sans-serif} .stDownloadButton button {min-height:2.5rem}</style>', unsafe_allow_html=True)
st.title('P57 · Indian traffic-sign detection')
st.caption('Local image detection · Baseline A and Degradation-aware C · Six supported classes')
page = st.radio('Workspace', ['Detect & Compare', 'Measured Research', 'Help / Limitations'], horizontal=True, label_visibility='collapsed')


@st.cache_resource
def worker_gate():
    return threading.Lock()


@st.cache_resource
def capabilities():
    return call_worker(ROOT, {'op':'capabilities'})


def worker(request):
    with worker_gate():
        return call_worker(ROOT, request)


def clear_results():
    for k in ['result', 'result_key', 'export_files', 'export_zip', 'export_key']:
        st.session_state.pop(k, None)


def research():
    st.subheader('Measured W6 research')
    st.info('C gains at O2/O3; A leads on clean, fading and O1. These are fixed benchmark measurements, not upload accuracy.')
    try:
        manifest = json.loads((ROOT/'app/research_manifest.json').read_text())
        directory = ROOT/manifest['root']
        for name, expected in manifest['files'].items():
            if sha((directory/name).read_bytes()) != expected:
                raise ValueError(f'Artifact authentication failed: {name}')
        rows = json.loads((directory/'paired_table.json').read_text())
        table = [{'Condition':r['condition'], 'A mAP50 (%)':100*r['A_mAP50'],
                  'C mAP50 (%)':100*r['C_mAP50'], 'C−A (pp)':100*r['delta_mAP50_fraction'],
                  'A precision (%)':100*r['A_precision'], 'C precision (%)':100*r['C_precision'],
                  'A recall (%)':100*r['A_recall'], 'C recall (%)':100*r['C_recall']} for r in rows]
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={key:st.column_config.NumberColumn(format='%.3f')
                                    for key in table[0] if key!='Condition'})
        st.caption('P/R use the evaluator’s smoothed best-F1 convention, not the upload display-confidence filter. Machine values remain fractions.')
        for name, title in [('fading_robustness.png','Fading'), ('occlusion_robustness.png','Occlusion')]:
            st.image(str(directory/name), caption=title, use_container_width=True)
        st.download_button('Download measured metrics JSON', (directory/'metrics_14.json').read_bytes(), 'w6_metrics.json', 'application/json', on_click='ignore')
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Measured research artifacts unavailable or failed authentication. No substitute results are shown. {error}')
    st.warning('Only 45 test sources; many are crops/illustrations. Upstream physical-sign, track and geographic provenance are unknown. Controlled masks are not all real-world degradation. GTSRB reference remains blocked; no significance, safety or universal robustness claim.')


if page == 'Measured Research':
    research()
elif page == 'Help / Limitations':
    st.subheader('How to use the local app')
    st.markdown('1. Upload a JPEG or PNG (up to 10 MiB and 12 megapixels).\n2. For simulation, mark one or more regions using the horizontal/vertical ranges, then **Add Region**.\n3. Select the input condition, model(s) and device, then **Run detection**.\n4. Adjust display confidence and save the images, JSON/CSV or results ZIP.')
    st.write('Uploads have no ground truth. Marked regions only control simulation; they are not labels or automatic sign segmentation. The exact same transformed pixels go to both models, sequentially.')
    st.write('EXIF orientation is applied first. Grayscale and CMYK are converted to RGB; transparency is composited on white. Metadata is stripped. No calibrated color-profile conversion is claimed. Animated, corrupt and oversized files are rejected.')
    st.write('CPU is the default. MPS is offered only when available and is never silently substituted. Preprocessing is fixed 448×448 centered letterbox with aspect ratio preserved; coordinates map back to the normalized upload. Predictor NMS is class-aware, single-label, IoU .7, confidence floor .001 and max_det 300, without a wall-clock break. W6 evaluator used multi-label matching and different padding; upload predictions are not benchmark metric reproduction.')
    st.write('Display confidence starts at .25. Changing it filters stored raw predictions identically for A/C; it does not rerun inference. A new image resets regions/results. Region, condition, model, device or model-manifest changes invalidate results. Downloads reuse prepared bytes and never run inference.')
    st.write('No upload mAP or accuracy is computed. No UI action can train, run the benchmark, install packages, download weights or publish. Inputs and exports stay in session memory; worker temporary directories are removed. Close the browser session to discard memory; downloading is an explicit local save.')
    st.write('Supported classes: '+', '.join(n.replace('_',' ').title() for n in CLASSES))
    st.caption('Dataset: public SDI Indian Traffic Sign v1, CC BY 4.0; reused annotations, student-reported faculty approval. W2/G05 reference, cross-machine reproduction and final submission remain incomplete. AI assisted implementation; team sign-off remains pending.')
else:
    st.subheader('Detect & Compare')
    try:
        caps = capabilities()
        manifest, manifest_sha = model_manifest(ROOT, os.environ.get('P57_MODEL_MANIFEST'))
    except Exception as error:
        st.error(str(error)); st.stop()
    with st.sidebar:
        st.header('Detection settings')
        model_choice = st.radio('Models', ['Side by side · A + C', 'Baseline A', 'Degradation-aware C'])
        device_choice = st.selectbox('Device', ['CPU']+(['Apple GPU / MPS'] if caps['mps_available'] else []))
        st.caption('CPU default. Apple GPU/MPS '+('available.' if caps['mps_available'] else 'unavailable; not offered.'))
        family = st.radio('Input condition', ['Original', 'Fading', 'Occlusion'])
        severity = st.select_slider('Simulation severity', ['Mild', 'Medium', 'Strong'], disabled=family=='Original')
        confidence = st.slider('Display confidence', 0.0, 1.0, .25, .01,
                               help='Filters stored predictions. No new inference when only this value changes.')
        st.caption('Raw inference floor .001 · NMS IoU .7 · fixed 448×448 · FP32')
    upload = st.file_uploader('Upload a road/sign image', type=['jpg','jpeg','png'], accept_multiple_files=False)
    if upload is None:
        if st.session_state.get('upload_sha'):
            st.session_state.pop('upload_sha',None); st.session_state['regions']=[]; st.session_state.pop('preview',None);clear_results()
        st.info('Choose an image to begin. Processing stays local; uploads never enter the research dataset.')
        st.stop()
    raw = upload.getvalue()
    upload_sha = sha(raw)
    if st.session_state.get('upload_sha') != upload_sha:
        st.session_state.update(upload_sha=upload_sha, regions=[])
        st.session_state.pop('preview',None); clear_results()
    try:
        original, normalization = normalize(raw)
    except ValueError as error:
        clear_results(); st.error(str(error)); st.stop()
    normalized = png(original)
    st.caption(f'Normalized RGB image: {original.width} × {original.height} pixels · EXIF orientation applied · original preserved')
    left, right = st.columns(2)
    with left:
        st.image(original, caption='Normalized original', use_container_width=True)
    with right:
        st.markdown('**User-marked simulation regions**')
        st.caption('Pixel ranges on the normalized image; no ground-truth or automatic segmentation claim.')
        # Image-specific keys reset controls when a different upload arrives.
        h_range = st.slider('Horizontal range (pixels)', 0, original.width, (0, original.width), key='h_'+upload_sha)
        v_range = st.slider('Vertical range (pixels)', 0, original.height, (0, original.height), key='v_'+upload_sha)
        if st.button('Add Region'):
            try:
                st.session_state['regions'] = regions_valid(st.session_state['regions']+[[h_range[0],v_range[0],h_range[1],v_range[1]]], *original.size)
                clear_results();st.session_state.pop('preview',None)
            except ValueError as error:
                st.error(str(error))
        if st.button('Reset regions'):
            st.session_state['regions']=[];clear_results();st.session_state.pop('preview',None)
        for i, region in enumerate(st.session_state['regions']):
            st.write(f'Region {i+1}: {region}')
            if st.button(f'Remove region {i+1}',key='remove_'+str(i)):
                st.session_state['regions'].pop(i);clear_results();st.session_state.pop('preview',None);st.rerun()
        if st.session_state['regions']:
            st.image(region_preview(original,st.session_state['regions']),caption='Orange rectangles are simulation regions only',use_container_width=True)
    regions = st.session_state['regions']
    models = ['A','C'] if model_choice.startswith('Side') else ['A'] if model_choice.startswith('Baseline') else ['C']
    device = 'mps' if device_choice.startswith('Apple') else 'cpu'
    condition = 'original' if family=='Original' else ('F' if family=='Fading' else 'O')+str(['Mild','Medium','Strong'].index(severity)+1)
    current_key = state_key(normalization['normalized_pixel_sha256'], regions, condition, models, device, manifest_sha)
    if st.session_state.get('result_key') != current_key:
        clear_results()
    if condition!='original' and not regions:
        clear_results();st.warning('Fading/occlusion requires at least one user-marked simulation region. Select ranges and press Add Region.');st.stop()
    preview_key = state_key(normalization['normalized_pixel_sha256'],regions,condition,[],'cpu',manifest_sha)
    request = {'op':'transform','image_png':base64.b64encode(normalized).decode(),
               'regions':regions,'condition':condition,'manifest_path':os.environ.get('P57_MODEL_MANIFEST')}
    if st.session_state.get('preview_key') != preview_key or 'preview' not in st.session_state:
        try:
            with st.spinner('Preparing the shared input locally…'):
                preview = worker(request)
            st.session_state.update(preview=preview,preview_key=preview_key)
        except Exception as error:
            clear_results();st.error(str(error));st.stop()
    preview = st.session_state['preview']
    shared_bytes = base64.b64decode(preview['shared_png'])
    st.image(shared_bytes,caption=f'Shared input · {condition} · identical pixels for all selected models',width=620)
    st.caption('Shared pixel SHA-256: '+preview['shared_pixel_sha256'])
    if st.button('Run detection', type='primary'):
        clear_results()
        try:
            with st.status('Processing locally: '+('A then C' if len(models)==2 else models[0]),expanded=True) as status:
                st.write('Verifying checkpoint bytes, then running sequential inference. Waiting includes native cleanup.')
                result = worker({**request,'op':'infer','models':models,'device':device})
                if result['shared_pixel_sha256'] != preview['shared_pixel_sha256']:
                    raise ValueError('Shared preview/inference pixels differ; result rejected.')
                status.update(label='Detection complete',state='complete',expanded=False)
            st.session_state.update(result=result,result_key=current_key)
        except Exception as error:
            clear_results();st.error(str(error))
    if st.session_state.get('result_key') == current_key:
        result = st.session_state['result']
        export_key = (current_key,confidence)
        if st.session_state.get('export_key') != export_key:
            files, archive = exports(normalized, shared_bytes, result, confidence, normalization)
            st.session_state.update(export_files=files,export_zip=archive,export_key=export_key)
        files = st.session_state['export_files']
        columns = st.columns(len(models))
        for col, model in zip(columns,models):
            with col:
                st.subheader(manifest['models'][model]['name'])
                st.image(files[f'annotated_{model}.png'],use_container_width=True)
                rows = displayed(result['models'][model]['raw_predictions'], confidence)
                if not rows:st.info('No detections at this display confidence.')
                else:st.dataframe([{'Sign':r['class_name'].replace('_',' '),'Confidence':r['confidence'],'Box xyxy':r['xyxy']} for r in rows],hide_index=True,use_container_width=True)
                st.download_button(f'Download {model} image',files[f'annotated_{model}.png'],f'annotated_{model}.png','image/png',on_click='ignore')
        st.success(f"Completed on {device.upper()} in {result['process']['elapsed_seconds_including_cleanup']:.2f}s including cleanup · no upload accuracy/mAP computed")
        st.download_button('Download results ZIP',st.session_state['export_zip'],'p57_results.zip','application/zip',on_click='ignore')
        st.download_button('Download predictions/settings JSON',files['predictions.json'],'predictions.json','application/json',on_click='ignore')
        st.download_button('Download raw/displayed CSV',files['predictions.csv'],'predictions.csv','text/csv',on_click='ignore')
    st.caption('Changing inference inputs invalidates results. Display confidence only re-filters stored predictions; downloads do not rerun detection.')
