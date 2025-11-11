<h2 class="overview">Overview</h2>
<p align="center">
  <img src="PathCLAST_overivew.png" 
       alt="PathCLAST Overview Diagram" 
       style="width:100%; max-width:800px; display:block; margin:auto; border:1px solid #ddd;">
</p>

<h2 class="requirements">Requirements</h2>
Please ensure that all the libraries below are successfully installed:

<ul class="requirements-list">
  <p>python == 3.9</p>
  <li>torch == 2.1.0</li>
  <li>CUDA == 12.1</li>
  <li>torchtoolbox == 0.1.8</li>
  <li>torch_geometric == 2.5</li>
  <li>scikit-image == 0.24.0</li>
  <li>scanpy</li>
  <li>rpy2 == 3.4.5</li>
</ul>

<h2 class="Datasets">Datasets</h2>
<ul class="Datasets">
  <li><a href="https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0" target="_blank" rel="noopener">
  Human Breast Cancer Block A Section 1 (IDC)
  </a></li>
  </a></li>
  <li><a href="https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_web_summary.html" target="_blank" rel="noopener">
  Human Breast Cancer Ductal Carcinoma (BCDC)
</a></li>
</a></li>
  <li><a href="https://github.com/almaan/her2st" target="_blank" rel="noopener">
  Human HER2-positive breast tumor (Her2ST)
</a></li>
  </a></li>
  <li><a href="http://research.libd.org/spatialLIBD/" target="_blank" rel="noopener">
  Human dorsolateral pre-frontal cortex (DLPFC)
</a></li>
</ul>


<h2 class="Run">Run PathCLAST</h2>
<ul class="Run">
<div
  class="code-box"
  style="
    position: relative;
    padding: 1em;    nbsp;
    min-width: 300px;  
    background: #f5f5f5;
    border-radius: 4px;
  "
>
  <pre><code id="code1">python train.py</code></pre>
</div>
</ul>
<h2>Getting started</h2>
<p>
  See <a href="Tutorial_Her2ST/tutorial_her2st.ipynb">Tutorials</a>
</p>
