(1) resonance.py will take the parameters of 2-delta system as input. You can manually desing the 2-delta system according the number of resonance you want using the condition from "Resonace Condition for 2-delta system" section     in the paper. But if you can visually find the configuration you want by using the interactive tool from interactive.py too.
    It will calculate the number of resonances within k<3 by itself and optimize the windows centered at resonances accordingly.
    After optimization (around 2-5 mins), it will generate a figure as a png and pdf.
    All the information during the optimization will be saved as a CSV in the same directory for analysis later on.

(2) result_analysis.py will analyse the existing CSV file accordinly and generate a figure.

(3) interactive.py will give u an interactive tool to play around and get a feeling of the the two 2-delta and 3-delta tranmission probabilites. 
    You can also use this to find postion parameters to get the number of resonances within k<3 as u wish. 
    You can also manipulate the strenght parameters but we primarily used condition alpha_1 = - alpha_2 from theory.
