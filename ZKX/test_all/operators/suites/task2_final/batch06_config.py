def expand_config(document):
    scales=[]
    for operator in document["operators"]:
        name=operator["operator_name"]
        for dtype in document["dtypes"]:
            for profile in operator["profiles"]:
                parameters={key:value for key,value in profile.items() if key not in ("suffix","order_of_magnitude")}
                if name=="kalman_filter":
                    count=profile["observation_count"]
                    inputs=[{"input_name":"observations","shape":[count],"element_count":count,"dtype":dtype,"role":"primary"}]
                    parameters.update({"f":1,"q":1,"alpha_sq":1,"h":1,"r":2,"output_count":2,"execution_dtype":"FP32_TASK_SEQUENCE"})
                else:
                    count=profile["count"]
                    inputs=[{"input_name":"x" if name=="cubic" else "feature_bundle","shape":[count],"element_count":count,"dtype":dtype,"role":"primary"}]
                    parameters["output_count"] = count if name=="cubic" else "dynamic_int64_coordinates"
                scales.append({"operator_name":name,"module_name":operator["module_name"],"backend":"not_applicable",
                    "scale_id":f"{name}_{dtype.lower()}_{profile['suffix']}","order_of_magnitude":profile["order_of_magnitude"],
                    "actual_elements":count,"inputs":inputs,"parameters":parameters})
    return scales
