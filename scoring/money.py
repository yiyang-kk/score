"""
Home Credit Python Scoring Library and Workflow
Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller, 
Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -*- coding: utf-8 -*-



def cba(credit_type, lift, lift2, integration_cost=0, yearly_fee=0):
    """
    Get cost benefit and maximum query price
    """
    if credit_type==1:
        contract_count=3098621.52661325
        conversion_rate=0.38
        hc_rr=0.17
        br=0.023
        ltp_bad=-36522.0
        ltp_good=1817.0
        scoring_rr=0.11
    elif credit_type==2:
        contract_count=422771.054533826
        conversion_rate=0.255
        hc_rr=0.27
        br=0.054
        ltp_bad=-186434.0
        ltp_good=42374.0
        scoring_rr=0.16
    elif credit_type==3:
        contract_count=132264.625808535
        conversion_rate=0.074
        hc_rr=0.11
        br=0.056
        ltp_bad=-53376.0
        ltp_good=12895.0
        scoring_rr=0.26
        
    app_count=contract_count/conversion_rate
    
    query_count=app_count*(1-hc_rr)
    
    ltp_contract=(1-br)*ltp_good+br*ltp_bad
  
    
    br_all=(1-scoring_rr)*br+scoring_rr*lift*br

    new_br=br_all/(1-scoring_rr+lift2*scoring_rr)
    new_ltp_contract=(1-new_br)*ltp_good+new_br*ltp_bad

    #print(ltp_contract, new_ltp_contract)
    
    cb= (new_ltp_contract - ltp_contract)*contract_count-integration_cost-yearly_fee
    
    max_price=((new_ltp_contract - ltp_contract)*contract_count-integration_cost-yearly_fee)/query_count

    return cb, max_price
