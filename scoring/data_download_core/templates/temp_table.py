TEMP_TABLE_TEMPLATE = """
create table {temp_table_name} pctfree 0 
partition by list ( dt_block)
( 
   
{partition_string}

)  as
select 
      bs.skp_credit_case,
      sv.skf_approval_process_head,
      sv.date_decision, 
      sv.skp_scoring_vector_attribute,
      sv.num_value, sv.text_value, sv.dtime_value,
      num_group_position_1, num_group_position_2,
      num_group_position_3,
      vi.dt_block as dt_block
from
      {base}                                 bs
      inner join
      owner_dwh.f_scoring_vector_tt          sv  on sv.skp_credit_case = bs.skp_credit_case
                                                and sv.date_decision >= date'{date_decision_start}'
                                                and sv.date_decision <= date'{date_decision_end}'
      inner join
      {schema}.{prefix}vector_item_names      vi  on vi.skp_scoring_vector_attribute = sv.skp_scoring_vector_attribute

"""
