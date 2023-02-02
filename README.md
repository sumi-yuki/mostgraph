# Mostgraph
## Mostgraph [Forced Oscillation Technique (FOT) machine sold in Japan] measurement results estimation by deep learning.
No reference range has yet been established for the results of Mostograph measurements.
Even if reference ranges were established, it would be too broad and difficult to interpret in clinical practice.
Our strategy is to determine respiratory normal subjects by pattern recognition of all measurement items using deep learning, rather than a single item in the Mostograph measurement.
This deep learning was trained to distinguish between respiratory normal subjects and patients with bronchial asthma/cough variant asthma(CVA), but we believe it can be used to identify respiratory disease patients in general.
> blockquote
Other respiratory diseases such as bacterial pneumonia, interstital pneumonia, etc. can be diagnosed by chest X-ray and COPD can be diagnosed by spirometry.
However, patients with bronchial asthma/CVA sometimes difficult to diagnose immediately in the doctor's office.
Asthma is a disease that improves and worsens, and patients do not necessarily present with wheezes or expiratory airflow limitation at the time of hospital visit.
Chest X-ray findings are usually normal, and no specific findings are found on blood examination.
Although some asthma patients show an increased FeNO (Fractional exhaled Nitric Oxide), some people with allergic predispositions have an increased FeNO even if they do not have asthma.
For an objective diagnosis of asthma, it is necessary to demonstrate the variable expiratory airflow limitation or airway hyperresponsiveness. To reveal variable expiratory airflow limitation, the patient have expiratory airflow limitation at the time of presentation and have a positive bronchodilator (BD) responsiveness (reversibility) test, excessive variation in lung function between visits, or excessive variability in peak flow (https://ginasthma.org). Airway hyperresponsiveness test is rarely performed in Japan because it requires an hour-long test with a doctor in attendance.
In reality, in many cases, the physician will make a comprehensive diagnosis and provide diagnostic treatment in order to alleviate the patient's suffering as soon as possible.
Since it is sometimes difficult for a non-specialist respiratory physician to make an on-the-spot diagnosis in an outpatient setting, the results of a Mostograph measurement can aid in that diagnosis.
The diagnostic accuracy of identification of respiratory normal subjects and patients with bronchial asthma/cough asthma by deep learning with this program is about 70%.
It was reported that there is a weak or absent correlations between resistance and reactance values measured by Mostgraph and age, height, and weight among men and women [https://www.sciencedirect.com/science/article/pii/S2212534516000022]. In fact, adding these data did not improve the diagnostic accuracy .
In addition, even if separate models were created for men and women inputting gender, the accuracy did not improve.
The results of the Mostograph measurements are reported as exhaled, inhaled, total, and delta for each item, with the total being approximately the average of the exhaled and inhaled measurements, and the delta being the difference between the exhaled and inhaled measurements. There was no difference in diagnostic accuracy between using all reported items and gender as inputs to the model, and using only exhalation and inhalation measurement as inputs.
Therefore, we used the single model for men and women, and input only measurement results during exhalation and inhalation for each item.

# モストグラフ
## モストグラフ（日本で販売されている強制オシレーション方肺機能検査装置）測定結果の深層学習による評価
モストグラフ測定結果の基準範囲は未だ設定されていません。
基準範囲が設定されたとしても、範囲が広すぎて臨床現場での解釈が困難になると思われます。
そこで我々はモストグラフ測定の単一項目ではなく、深層学習による全ての測定項目のパターン認識で、呼吸器系正常者が判定出来ないかと考えました。
この深層学習は、呼吸器系正常者と、気管支喘息/咳喘息の患者を区別するように訓練されましたが、一般的に呼吸器疾患患者を識別するのに使用出来ると思います。
細菌性肺炎、間質性肺炎などの他の呼吸器疾患は胸部X,CT線で診断でき、COPDはスパイロメトリーで診断できます。
ただ、気管支喘息/咳喘息患者は、外来ですぐに診断することが難しい場合があります。
喘息は良くなったり悪くなったりする疾患であり、来院時に必ずしも聴診所見や肺機能検査で呼気気流制限を呈するわけではありません。
胸部X線検査所見は正常であり、採血でも特異的な所見はありません。
呼気一酸化窒素濃度が上昇する喘息患者もいますが一部の患者のみであり、アレルギー体質があると喘息が無くても上昇する人もいます。
他覚的診断のためには変動する呼気気流制限を明らかにするか、気道過敏性があることを明らかにする必要があります。変動する呼気気流制限を明らかにするためには、来院時に閉塞性換気障害を呈していて吸入改善試験が陽性であるか、来院するごとに一秒量が変動する、家でのピークフローメーターによるピークフローが変動する必要があります(https://ginasthma.org)。気道過敏性検査は医者がつきっきりで1時間かかる検査のため日本ではほとんど行われていません。
現実には患者さんの苦痛を早急に緩和するため、医師が総合的に診断して診断的治療を行う場合も多いです。
呼吸器専門医でない場合は外来でその場での診断が困難な場合もあるため、モストグラフ測定結果が、その診断の一助になりえます。
本プログラムによる深層学習による呼吸器系正常者と、気管支喘息/咳喘息の患者識別正解率は約7割です。
年齢、身長、体重とモストグラフ測定結果の相関は少ないとされており、実際に深層学習モデル検討の際に年齢、身長、体重を入力として入れても正解率は変わりませんでした。
また男女別のモデルを別に作っても、同じモデルで入力として性別を入力して判定しても正解率は変わりませんでした。
モストグラフ測定結果はそれぞれの項目について、呼気時、吸気時、総合、デルタが報告されますが、総合はほぼ呼気時と吸気時の平均であり、デルタは呼気時から吸気時の測定値を引いた差です。モデルの入力として報告される全ての項目と呼気時から吸気時として性別を入力して判定しても正解率は変わりませんでした。
そのため、男女同一のモデルとし、測定結果の入力はそれぞれの項目について呼気時と吸気時のみと致しました。
