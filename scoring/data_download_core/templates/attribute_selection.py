ATTRIBUTE_SELECTION_TEMPLATE = """

select t.*,
       case when t.cnt_bod1 > t.cnt_bod2 then '1st BoD'
            when t.cnt_bod2 > t.cnt_bod1 then '2nd BoD'
            else 'hard to say'
       end as scoring_phase
  from (select d.skp_scoring_vector_attribute,
               d.text_vector_attr_full_path,
               d.flag_array_datatype as flag_array,
               d.name_vector_attr_datatype,
              
               count(case when d.dist_vals > 0 then 1 else null end) cnt_cred_with_val,
               count(distinct d.skp_credit_case) as cnt_cred_all,
               max(d.dist_vals) max_dist_vals,
              
               sum(case when scoring_phase = 1 then 1 else 0 end) as cnt_bod1,
               sum(case when scoring_phase = 2 then 1 else 0 end) as cnt_bod2
          from (select bs.skp_credit_case,
                       va.skp_scoring_vector_attribute,
                       va.text_vector_attr_full_path,
                       va.flag_array_datatype,
                       va.name_vector_attr_datatype,
                      
                       count(distinct 
                                 case when sv.text_value <> 'XNA' then sv.text_value
                                      when sv.num_value is not null then cast(round(sv.num_value,2) as varchar2(32))
                                      when sv.dtime_value <> date'3000-01-01' then cast(sv.dtime_value as varchar2(32))
                                      else null
                                 end) as dist_vals,
                      
                      /* scoring phases encoding:           */
                      /* 1 ... OFFER (alternative offers)   */
                      /* 2 ... MAIN (2nd BoD)               */
                      /* 3 ... PRELIM (1st BoD)             */
                      /* 4 ... CLIENT                       */              
                      min(decode(ap.skp_scoring_phase, 4, 3, 3, 1, 2, 2, 4)) as scoring_phase               
                from {base}                                               bs
                     inner join owner_dwh.f_scoring_vector_tt             sv  
                             on sv.skp_credit_case = bs.skp_credit_case
                            and sv.date_decision >= date'{attribute_selection_date_decision_start}' 
                            and sv.date_decision <= date'{attribute_selection_date_decision_end}'
                     inner join owner_dwh.f_approval_process_head_tt      ap 
                             on ap.skp_credit_case = bs.skp_credit_case
                            and ap.skf_approval_process_head = sv.skf_approval_process_head
                            and ap.date_decision >= date'{attribute_selection_date_decision_start}' 
                            and ap.date_decision <= date'{attribute_selection_date_decision_end}'
                            and ap.date_decision = sv.date_decision -- matching partitions on each other
                            and ap.skp_credit_type = sv.skp_credit_type -- matching partitions on each other
                            and ap.skp_scoring_phase <> 1
                     inner join owner_dwh.dc_scoring_vector_attribute     va  
                             on va.skp_scoring_vector_attribute = sv.skp_scoring_vector_attribute
                -- select how many records should be included - last month is probably enough
               where bs.date_decision >= date'{attribute_selection_date_decision_start}' 
                 and bs.date_decision <= date'{attribute_selection_date_decision_end}'
               group by bs.skp_credit_case,
                        va.skp_scoring_vector_attribute,
                        va.text_vector_attr_full_path,
                        va.flag_array_datatype,
                        va.name_vector_attr_datatype
              )                                                 d
        group by d.skp_scoring_vector_attribute, 
                 d.text_vector_attr_full_path,
                 d.flag_array_datatype,
                 d.name_vector_attr_datatype
      )                                                         t
{where_condition}

"""
