def expand_config(document):
    scales = []
    for operator in document["operators"]:
        name = operator["operator_name"]
        for dtype in document["dtypes"]:
            for profile in operator["profiles"]:
                parameters = dict(profile)
                parameters.pop("suffix")
                parameters.pop("order_of_magnitude")
                if name == "fm_demod":
                    count = profile["count"]
                    inputs = [{"input_name":"analytic","shape":[count],"element_count":count,"dtype":"Complex"+dtype,"role":"primary"}]
                    output_count = count - 1
                elif name == "correlate":
                    count = profile["signal_count"]
                    reference_count = profile["reference_count"]
                    inputs = [
                        {"input_name":"signal","shape":[count],"element_count":count,"dtype":dtype,"role":"primary"},
                        {"input_name":"reference","shape":[reference_count],"element_count":reference_count,"dtype":dtype,"role":"secondary"},
                    ]
                    output_count = count
                else:
                    count = profile["count"]
                    frame = profile["frame"]
                    frames = 1 + (count - frame) // profile["hop"]
                    inputs = [{"input_name":"signal","shape":[count],"element_count":count,"dtype":dtype,"role":"primary"}]
                    output_count = frame * frames
                parameters["output_count"] = output_count
                scales.append({
                    "operator_name": name,
                    "module_name": operator["module_name"],
                    "backend": operator["backend"],
                    "scale_id": f"{name}_{dtype.lower()}_{profile['suffix']}",
                    "order_of_magnitude": profile["order_of_magnitude"],
                    "actual_elements": sum(item["element_count"] for item in inputs),
                    "inputs": inputs,
                    "parameters": parameters,
                })
    return scales
