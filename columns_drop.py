import pandas as pd

columns_to_drop_loan = [
    'trm_amnt', 
    'trm_dsbsl_amnt', 
    'trm_dsbsl_curncy', 
    'trm_dsbsl_time', 
    'trm_liab_curexch', 
    'trm_liab_curexch_rate', 
    'trm_liab_nonpay', 
    'delnq', 
    'img_id', 
    'img_tmpl_id', 
    'vid_id', 
    'vdd_utb_id', 
    'brwr_pic', 
    'brwr_nopic', 
    'jrnl_cnt', 
    'jrnlblk_cnt', 
    'amnt_basket', 
    'amnt_curloss', 
    'time_posted_order', 
    'name'
]

columns_to_drop_lender = [
    'inviter', 
    'invitee_cnt', 
    'img_id', 
    'img_tmpl_id', 
    'url', 
    'name', 
    'occu_info', 
    'loan_bcuz'
]

loaners = pd.read_csv('KivaCSVs/05_L_loan_L.csv')
loaners_new = loaners.drop(columns=columns_to_drop_loan)

loaners_new.to_csv('05_L_Loan_L.csv', index=False)