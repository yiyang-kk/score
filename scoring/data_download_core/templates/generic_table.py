GENERIC_TABLE_CREATE_TEMPLATE = """
create table {table_name} as
select
    *
from
    (
        select
        1 as skp_credit_case,
        {num_group_creation_string}
        cast('a' as varchar2(256)) as val,
        vin.final_colname
    from
        {prefix}vector_item_names            vin
    where vin.dt_block = '{dt_block}'            
      ) pivot(max(val) for final_colname in ( 
          {variable_string}
))
"""
# ;

# delete from jh_data_vector_dyn_persons_1;
# commit;

GENERIC_TABLE_INSERT_TEMPLATE = """
insert /*+append*/ into {table_name}
(
select
    *
from
    (
        select
        t.skp_credit_case,
        {num_group_insert_string_1}
        case
                  when t.num_value is not null              then cast(round(t.num_value,5) as varchar2(256))
                  when t.text_value <> 'XNA'                then cast(t.text_value as varchar2(256))
                  when t.dtime_value <> date'3000-01-01'    then to_char(t.dtime_value, 'YYYY-MM-DD HH24:MI:SS')
              end as val,
        t.final_colname
    from
        (
            select /*+parallel(4)*/
                bs.skp_credit_case,

                sv.text_value,
                sv.num_value,
                sv.dtime_value,

                {num_group_insert_string_2}
                va.skp_scoring_vector_attribute,
                va.final_colname,

                /* scoring phases encoding:           */
                /* 1 ... OFFER (alternative offers)   */
                /* 2 ... MAIN (2nd BoD)               */
                /* 3 ... PRELIM (1st BoD)             */
                /* 4 ... CLIENT                       */
                decode(ap.skp_scoring_phase, 4, 1, 3, 3, 2, 2, 0) as scoring_phase,
                max(decode(ap.skp_scoring_phase, 4, 1, 3, 3, 2, 2, 0)) over (partition by sv.skp_credit_case, sv.skp_scoring_vector_attribute) as last_scoring_phase
            from
                {base}                              bs
                inner join
                {temp_table}                        sv 
                    on sv.skp_credit_case = bs.skp_credit_case
                    and sv.date_decision >= date'{date_decision_start}' -- for partition pruning
                    and sv.dt_block = '{dt_block}' -- use previously defined partitioning
                inner join
                (
                    select
                        vin.skp_scoring_vector_attribute,
                        vin.final_colname
                    from
                        {prefix}vector_item_names  vin
                    where vin.dt_block = '{dt_block}'
                )                                   va 
                on va.skp_scoring_vector_attribute = sv.skp_scoring_vector_attribute
                inner join
                owner_dwh.f_approval_process_head_tt     ap 
                    on ap.skp_credit_case = bs.skp_credit_case
                    and ap.skf_approval_process_head = sv.skf_approval_process_head
                    and ap.date_decision >= date'{date_decision_start}' -- for partition pruning
                    and ap.date_decision = sv.date_decision -- matching partitions on each other
                    and ap.skp_scoring_phase <> 1
            -- exclude OFFER scoring phase (alternative offer - this is generated after score application)
            where 1 = 1
        )                                                t
    where t.scoring_phase = last_scoring_phase
      )
pivot (max(val) for final_colname in (
    {variable_string}
           )))
"""
