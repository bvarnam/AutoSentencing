# Problem Statement
It is well known that people make judgements on others based on demographic information like race, sex, and education. This phenomenon is already quite well documented in the justice system, with many think pieces and academic texts alike decrying the effects that these factors have on criminal defendants. ****This project is an exploration of whether such judgements affect whether or not criminal defendants in drug possession and drug trafficking cases.****  We plan on creating and training various classification models on federal sentencing data. One set of models will include demographic information, and one set will not. If the set with demographic information performs substantially better (has about a 10% lower misclassification rate) than the set without, we will likely have discovered some demographic features that unduly influence criminal sentencing. Although all of the models that included demographic information performed slightly better than those that did not, the difference was not large enough to definitively say that demographic information has predictive power in federal sentencing. An executive summary of our project can be found [here](https://github.com/123CarlosRivera/project5/blob/main/EXEC.md).

# Contents

 - `assests` contains image files for the visualizations used in our slide deck.
 - - `app` contains the webapp detailed in the executive summary
 - `code` contains jupyter notebooks and python scripts used to create and train models, as well as aggregate and explore data.
 - `data` contains, as its name suggest, data files in csv format. All models were trained using data from `drugs_2020_simply_imputed.csv` 
 - `models` contains saved or pickled versions of the two neural nets we created as well as a logistic regression model.
 
# Software Packages
The following python libraries were used in this project:

 - `Pandas` for data ingestion and cleaning
 - `scikitlearn` for model preprocessing, selection, and training
 - `numpy` for general mathematical operations
 - `tensorflow` and `keras` for neural net creating and training

# Data
All of the models created in thes project were trained on information made publicly available by the United States Sentencing Comission on [its website](https://www.ussc.gov/research/datafiles/commission-datafiles#individual). The script provided to make SAS files from the raw data files might be broken. 

After reading these data into a pandas dataframe, about 64 initial features were selected based on what we thought would predict whether a defendant received a prison sentence. Null values were then either dropped or imputed with the mean for numeric features and the mode for categorical or ordinal features.

## Data Dictionary
A more in-depth guide to all the data available in these datasets can be found [here](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf).

|Dataset |Feature  | Description| Datatype|
--- | --- | --- | ---
|drugs_2020_simply_imputed.csv|accgdln|"Explicit statement by the Court on the Statement of Reasons (SOR) regarding acceptance of the guideline factors applied in the PSR."|float64
|drugs_2020_simply_imputed.csv|age|Age of defendant at time of sentencing|float64
|drugs_2020_simply_imputed.csv|altdum|Indicator of whether defendant was given an alternative sentence (1 for yes, 0 for no)|int64
|drugs_2020_simply_imputed.csv|amttotal|Total amount of money defendant was ordered to pay, including fines, cost of supervision, and restitution (0 if no financial penalty imposed|int64
|drugs_2020_simply_imputed.csv|casetype|Type of case (1: Felony, 2: Misdemeanor A, 3: Misdemeanor B/C)|float64
|drugs_2020_simply_imputed.csv|citwhere|Defendant's country of citizenship. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for a full list of country codes|float64
|drugs_2020_simply_imputed.csv|combdrg2|Drug type (1: Cocaine, 2: Crack, 3: Heroin, 4: Marijuana, 6: Methamphetamine, 77: Other)|float64
|drugs_2020_simply_imputed.csv|crimhist|0: Defendant has no criminal history, 1: defendant has criminal history|float64
|drugs_2020_simply_imputed.csv|disposit|Disposition of defendant's case (0: No imprisonment, 1: Guilty Plea, 2: Nolo Contendere, 3: Jury Trial, 4: Trial by Judge or Bench Trial, 5: Guilty Plea and Trial (>1 count)|int64
|drugs_2020_simply_imputed.csv|district|District in which defendant was sentenced. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for full list of districts|int64
|drugs_2020_simply_imputed.csv|drugmin|Mandatory minimum sentence (in months) for particular drug statutes|int64
|drugs_2020_simply_imputed.csv|dsplea|Status indicator for Plea Agreement document (0: Not received, 1: Received, 2: Received alternate document, 3: Oral plea agreement, 5: Straight up Plea, No Agreement, 8: Trial, 9: Guilty Plea of indeterminable type)|int64
|drugs_2020_simply_imputed.csv|educatn|Highest level of education obtained by defendant|float64
|drugs_2020_simply_imputed.csv|intdum|Whether or not a defendant received intermittent confinement (0: No, 1: yes)|int64
|drugs_2020_simply_imputed.csv|methmin|Mandatory minimum sentence (in months) associated with a methamphetamine-related offense (see [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for particular statutes|int64
|drugs_2020_simply_imputed.csv|monrace|Offender's race (self-reported) (1: White/Caucasian, 2: Black/African American, 3: American Indian/Alaskan Native, 4: Asian or Pacific Islander, 5: Multi-racial, 7: Other, 8: Info not available, 9: Non=US American Indians, 10: American Indians Citizenship Unknown)|int64
|drugs_2020_simply_imputed.csv|monsex|Offender's sex (0: Male, 1: Female)|float64
|drugs_2020_simply_imputed.csv|mweight|Marijuana weight equivalency in grams of all drugs coded|float64
|drugs_2020_simply_imputed.csv|newcit|Citizenship of defendant (1: US, 1: Non U.S.)|float64
|drugs_2020_simply_imputed.csv|newcnvtn|Whether case was settled by plea agreement or trial (0: Plea, 1: Trial)|int64
|drugs_2020_simply_imputed.csv|neweduc|Highest level of education obtained by offender (1: Less than H.S. Graduate, 3: H.S. Graduate, 5: Some College, 6: College Graduate)|float64
|drugs_2020_simply_imputed.csv|newrace|Race of defendant (1: White, 2: Black, 3: Hispanic, 6: Other)|float64
|drugs_2020_simply_imputed.csv|nodrug|Number of drugs involved in a case|float64
|drugs_2020_simply_imputed.csv|numdepen|Number of dependents supported by the defendant (excluding self)|float64
|drugs_2020_simply_imputed.csv|offguide|Primary type of crime under discussion. See (see [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for particular more information|int64
|drugs_2020_simply_imputed.csv|prisdum|Whether the defendant received a prison sentence (0: No, 1: Yes)|int64
|drugs_2020_simply_imputed.csv|probatn|Total probation ordered in months|float64
|drugs_2020_simply_imputed.csv|probdum|Whether or not a defendant received probation (0: No, 1: Yes)|int64
|drugs_2020_simply_imputed.csv|quarter|Quarter of the fiscal year in which case was sentenced (1: Oct 1-Dec 31, 2: Jan 1-Mar 31, 3: Apr 1-Jun 30, 4: July 1-Sept 30)|int64
|drugs_2020_simply_imputed.csv|reas1|The first reason given by the court for why a sentence imposed was outside of the recommended range. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for list of reasons|float64|
drugs_2020_simply_imputed.csv|reas2|The second reason given by the court for why a sentence imposed was outside of the recommended range. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for list of reasons|float64
|drugs_2020_simply_imputed.csv|reas3|The third reason given by the court for why a sentence imposed was outside of the recommended range. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for list of reasons|float64
|drugs_2020_simply_imputed.csv|regsxmin|Mandatory minimum sentence in months regarding failure to register as a sex offender|int64
|drugs_2020_simply_imputed.csv|relmin|Mandatory minimum sentence in months for offenses committed while on release (see [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for more details)|int64
|drugs_2020_simply_imputed.csv|resdet1|Court's determination of restitution (1: Not ordered because number of victims too large, 2: Not ordered because of complex issues of fact, 3: Not ordered because the complication and prolongation of sentencing process from fashioning order outweighs the need to provide restitution to victims, 4: Not ordered for other reasons, 5: Partial restitution ordered, 6: Not ordered because victims' losses were not ascertainable, 7: Not ordered because victims elected not to go through the restitution determining phase)|float64
|drugs_2020_simply_imputed.csv|restdum|Whether restitution amount was given (0: No, 1: Yes)|float64
|drugs_2020_simply_imputed.csv|safe|Indicator of level of safety valve application. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for specifics.|float64
|drugs_2020_simply_imputed.csv|safety|Indicates whether safety valve provision was applied (0: No, 1: Yes)|float64
|drugs_2020_simply_imputed.csv|senspcap|Total prison sentence in months plus alternatives (includes zeroes for probation)|float64
|drugs_2020_simply_imputed.csv|sensplt0|Total prison sentence, in months, plus alternatives (includes zeros for probtion) under a different statute.|float64
|drugs_2020_simply_imputed.csv|sentimp|Indicates type of sentence given (0: No prison/Probation (Fine Only), 1: Prison Only, 2: Prison + Confinement Conditions, 3: Probation + Confinement Conditions, 4: Probation Only)|int64
|drugs_2020_simply_imputed.csv|smax1|Statutory maximum for first count of conviction|float64
|drugs_2020_simply_imputed.csv|smin1|Statutory minimum for first count of conviction|float64
|drugs_2020_simply_imputed.csv|sources|Indicates extent to which information coded in guideline application is based upon *known* court findings. (1: Information represents known court findings, 2: Alternate Docs/some info may be missing from SOR, 3: PSR is Coded (Insufficient info for SOR), 5: PSR is Coded (No SOR present), 6: 18§924(c) only, 7: 8§1028A only, 8: No analagous guidelines, 9: PSR Waived, missing, or multiple offense levels). Code 2 was discontinued in FY2008.|int64|
|drugs_2020_simply_imputed.csv|statmax|Statutory maximum prison term, in months, for al prison counts|float64
|drugs_2020_simply_imputed.csv|statmin|Statutory minimum prison term, in months, for all counts|float64
|drugs_2020_simply_imputed.csv|supermax|Maximum guideline range (in months) for supervised release as determined by the court|float64
|drugs_2020_simply_imputed.csv|supermin|Minimum guideline range (in months) for supervised release as determined by the court|float64
|drugs_2020_simply_imputed.csv|suprdum|Whether defendant received supervised release (0: No, 1: Yes)|int64
|drugs_2020_simply_imputed.csv|suprel| Number of months of supervised release ordered by the court|int64
|drugs_2020_simply_imputed.csv|timservc|Number of months of time served credited to the offender by the judge at the time of sentencing|float64
|drugs_2020_simply_imputed.csv|totchpts|Total number of criminal history appoints applied|float64
|drugs_2020_simply_imputed.csv|totrest|Total dollar amount of restitution ordered|float64
|drugs_2020_simply_imputed.csv|totunit|Number of units used to calculate offense level adjustment to be applied in cases of multiple counts or grouping adjustments|float64
|drugs_2020_simply_imputed.csv|typemony|Indicates whether a fine/cost of supervision or restitution was ordered (1: Neither, 2: Restitution only, 3: Fine/cost of supervision only, 4: Both)|float64
|drugs_2020_simply_imputed.csv|typeoths|Other types of sentences imposed (0: None, 1: Suspended prison term, 2: Pay cost of prosecution, 3: Denial of federal benefits, 77: Other)|int64
|drugs_2020_simply_imputed.csv|unit1|Unit of measure for the first drug type (1: g, 2: kg, 3: lb, 4: oz, 5: Plant, 6: Dose, 7: gal, 8: qt, 9: L, 10: mg, 11: ml, 13: Marijuana cigarette, 77: Other)|float64
|drugs_2020_simply_imputed.csv|mwgt1|Marijuana equivalency weight in grams of first drug type|float64
|drugs_2020_simply_imputed.csv|Weight in grams of first drug type for which the defendant was held responsible| float64
|drugs_2020_simply_imputed.csv|xcrhissr|Defendant's criminal history category (I-VI)|float64
|drugs_2020_simply_imputed.csv|First offense level|float64
|drugs_2020_simply_imputed.csv|xmaxsor|Maximum guideline sentencing range determined by the court|float64
|drugs_2020_simply_imputed.csv|xminsor|Minimum guideline sentencing range determined by the court|float64
|drugs_2020_simply_imputed.csv|sentrnge|Assigns cases to one of 8 post-booker reporting categories based on relationship between the sentence and guideline range. See [codebook](https://www.ussc.gov/sites/default/files/pdf/research-and-publications/datafiles/USSC_Public_Release_Codebook_FY99_FY20.pdf) for specific categories.|float64

All of the above features are also present in `aggregate_years.csv`. 

Only the following features were used in our models:
Non-demographic features:
`'accgdln', 'altdum', 'amttotal', 'casetype', 'citwhere', 'combdrg2', 'crimhist', 'disposit', 'district', 'drugmin', 'dsplea', 'intdum', 'methmin', 'mweight', 'nodrug', 'numdepen', 'offguide', 'probatn', 'probdum', 'quarter', 'reas1', 'reas2', 'reas3', 'smax1', 'smin1', 'sources', 'statmax', 'statmin', 'supermax', 'supermin', 'suprdum', 'suprel'`

Demographic features:
`'age', 'newrace', 'monsex', 'monrace', 'neweduc', 'newcnvtn', 'educatn'`
