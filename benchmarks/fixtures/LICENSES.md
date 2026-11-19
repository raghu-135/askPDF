# Bundled benchmark PDFs

Redistribution is allowed under the license named per paper. Full PDF bytes live in
`bundled/`. Optional arXiv fixtures remain manifest-only (see `manifest.json`).

| File | Title | Authors | Source | License |
| --- | --- | --- | --- | --- |
| `plos-pone-0001208.pdf` | Fast Economic Development Accelerates Biological Invasions in China | Wen Lin et al. | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0001208 | CC BY 4.0 |
| `plos-pcbi-1004961.pdf` | Ten Simple Rules for Effective Statistical Practice | Robert E. Kass et al. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004961 | CC BY 4.0 |
| `plos-biology-0020391.pdf` | Widespread A-to-I RNA Editing of Alu-Containing mRNAs in the Human Transcriptome | Athanasiadis et al. | https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0020391 | CC BY 4.0 |
| `plos-medicine-0030399.pdf` | How Did Social Medicine Evolve, and Where Is It Heading? | Dorothy Porter | https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0030399 | CC BY 4.0 |
| `plos-pathogens-1008410.pdf` | Novel phenotype of Wolbachia strain wPip in Aedes aegypti… | Fraser et al. | https://journals.plos.org/plospathogens/article?id=10.1371/journal.ppat.1008410 | CC BY 4.0 |
| `plos-genetics-1003700.pdf` | A Highly Redundant Gene Network Controls Assembly of the Outer Spore Wall in S. cerevisiae | Lin et al. | https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1003700 | CC BY 4.0 |

Eval harness uses only the bundled files above. Optional arXiv downloads (`manifest.json`) are
for legacy POC scripts only — not redistributed and not used by `rag_eval_suite.py`.

## IR corpus PDFs (generated)

| File | Source | License | Notes |
| --- | --- | --- | --- |
| `ir-nfcorpus-bundle-01.pdf` | [BEIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | **CC BY-SA 4.0** | NFCorpus license (nutrition/medical abstracts; BEIR metadata). Share-alike applies to derivatives. |
| `ir-nfcorpus-bundle-02.pdf` | BEIR/nfcorpus | **CC BY-SA 4.0** | Same as above. |
| `ir-trec-covid-bundle-01.pdf` | [BEIR/trec-covid](https://huggingface.co/datasets/BeIR/trec-covid) / [CORD-19](https://allenai.org/data/cord-19) | **Heterogeneous (CORD-19)** | Abstract text from CORD-19; underlying articles vary (PMC OA, publisher terms). Bundles use BEIR-selected passages only. |
| `ir-trec-covid-bundle-02.pdf` | BEIR/trec-covid / CORD-19 | **Heterogeneous (CORD-19)** | Same as above. |

### Excluded (not redistributed in this repo)

| Source | License | Reason |
| --- | --- | --- |
| BEIR **SciFact** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (per [BEIR dataset card](https://huggingface.co/datasets/BeIR/scifact) and SciFact repository) | Non-commercial restriction; incompatible with committing full text PDFs here. |