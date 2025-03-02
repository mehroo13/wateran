File "/mount/src/streamflow-prediction/streamlit_app.py", line 130, in <module>
    input_vars = st.multiselect("🔍 Input Variables", [col for col in numeric_cols if col != output_var], default=[numeric_cols[0]])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/metrics_util.py", line 410, in wrapped_func
    result = non_optional_func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/widgets/multiselect.py", line 229, in multiselect
    return self._multiselect(
           ^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/widgets/multiselect.py", line 277, in _multiselect
    default_values = get_default_indices(indexable_options, default)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/lib/options_selector_utils.py", line 91, in get_default_indices
    default_indices = check_and_convert_to_indices(indexable_options, default)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/lib/options_selector_utils.py", line 74, in check_and_convert_to_indices
    raise StreamlitAPIException(
