# paper_2024_covid_cough
Our study addresses the pressing need for robust algorithms in physiological time series analysis by introducing a novel approach for COVID-19 detection through cough audio recordings

Abstract:
In March 2020, the World Health Organization declared COVID-19 to be a global pandemic. To date it has caused the deaths of over 6 million people worldwide. Gold standard methods for diagnosis of the disease remain challenging for pandemic control due to the need for a physical examination, which cannot be performed remotely. Biomedical research in COVID-19 detection using cough audio signals has gained significance due to the potential of publicly available datasets to provide a vast amount of data for the development and improvement of AI screening tools and deep learning algorithms. This study presents a signal processing system for COVID-19 detection based on participants’ cough audio recordings. Our main objective was to investigate the system’s ability to identify COVID-19 and make performance comparison to conventional AI-based methods. The system involves cough event detection and segmentation, followed by COVID-19 classification from the detected cough events using a Recurrent Convolutional Neural Network. Audio recordings were collected from 664 participants using publicly available datasets acquired via mobile platforms. We defined exclusion criteria to overcome the limitations of current datasets. The detection of cough events from the audio recordings yielded an UAR and AUC of 92.5% and 98.2% respectively. For COVID-19 classification, this approach achieved an UAR and AUC of 71.7% and 77.4% respectively. The results of the study emphasize the significant possibility of utilizing cough-based signal processing systems for COVID-19 diagnosis. Such a method of detection is rapid, user-friendly, non-intrusive, and can be performed remotely, which is essential during pandemic situations.

![proposed system simple ver3](https://github.com/chikleWhite/paper_2024_covid_cough/assets/136971404/3ef51277-5d82-4a08-a20e-dbdd9ad62357)

Cough event detection:
We implemented a cough event detection system (Fig. 5) using MobileNet-based architecture named Yet Another Mobile Network (YAMNet).

![image](https://github.com/chikleWhite/paper_2024_covid_cough/assets/136971404/b40f29f9-2ceb-42b5-99f0-655349f01bc8)
![image](https://github.com/chikleWhite/paper_2024_covid_cough/assets/136971404/ab3e1fd6-5f55-4661-a7c6-e44702e256af)

COVID-19 Classification:
![image](https://github.com/chikleWhite/paper_2024_covid_cough/assets/136971404/143e57a3-4bfa-4376-a02b-566fbe384062)
