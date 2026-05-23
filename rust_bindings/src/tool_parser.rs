use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
pub struct ToolCallStreamParser {
    state: u8,
    buf: String,
    current_index: usize,
    emitted_calls: usize,
}

#[pymethods]
impl ToolCallStreamParser {
    #[new]
    fn new() -> Self {
        ToolCallStreamParser {
            state: 0,
            buf: String::new(),
            current_index: 0,
            emitted_calls: 0,
        }
    }

    fn process<'py>(&mut self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        let mut results = PyList::empty_bound(py);

        if self.state == 0 {
            self.buf.push_str(text);
            if let Some(begin_idx) = self.buf.find("<|tool_calls_section_begin|>") {
                let before = &self.buf[..begin_idx];
                if !before.is_empty() {
                    results.append(("content", before))?;
                }
                self.state = 1;
                let rest = self.buf[begin_idx + "<|tool_calls_section_begin|>".len()..].to_string();
                self.buf = rest;
                self._process_buffer(py, &mut results)?;
            } else if !self.buf.contains("<|tool") && self.buf.len() > 30 {
                results.append(("content", &self.buf))?;
                self.buf.clear();
            }
        } else if self.state == 1 {
            self.buf.push_str(text);
            if let Some(end_idx) = self.buf.find("<|tool_calls_section_end|>") {
                let remaining = self.buf[..end_idx].to_string();
                self.buf = remaining;
                self._process_buffer(py, &mut results)?;
                results.append(("tool_call_end", py.None()))?;
                self.state = 2;
                self.buf.clear();
            } else {
                self._process_buffer(py, &mut results)?;
            }
        }

        Ok(results)
    }

    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut results = PyList::empty_bound(py);
        if self.state == 0 && !self.buf.is_empty() {
            results.append(("content", &self.buf))?;
            self.buf.clear();
        } else if self.state == 1 {
            self._process_buffer(py, &mut results)?;
            if self.emitted_calls > 0 {
                results.append(("tool_call_end", py.None()))?;
            }
        }
        Ok(results)
    }
}

impl ToolCallStreamParser {
    fn _process_buffer<'py>(&mut self, py: Python<'py>, results: &mut Bound<'py, PyList>) -> PyResult<()> {
        while let Some(begin_idx) = self.buf.find("<|tool_call_begin|>") {
            if let Some(end_idx) = self.buf.find("<|tool_call_end|>") {
                // Find arguments
                let search_area = &self.buf[begin_idx..end_idx];
                if let Some(arg_begin_rel) = search_area.find("<|tool_call_argument_begin|>") {
                    let func_part = &search_area["<|tool_call_begin|>".len()..arg_begin_rel];

                    if func_part.starts_with("functions.") {
                        if let Some(colon_idx) = func_part.find(':') {
                            let name = &func_part["functions.".len()..colon_idx];
                            let index_str = &func_part[colon_idx + 1..];
                            if let Ok(index) = index_str.parse::<usize>() {
                                let arguments = search_area[arg_begin_rel + "<|tool_call_argument_begin|>".len()..].trim();

                                let uuid = uuid::Uuid::new_v4().simple().to_string();
                                let call_id = format!("call_{}", &uuid[..8]);

                                let dict = PyDict::new_bound(py);
                                dict.set_item("index", index)?;
                                dict.set_item("id", call_id)?;
                                dict.set_item("type", "function")?;

                                let func_dict = PyDict::new_bound(py);
                                func_dict.set_item("name", name)?;
                                func_dict.set_item("arguments", "")?;
                                dict.set_item("function", func_dict)?;

                                results.append(("tool_call_start", dict))?;

                                if !arguments.is_empty() {
                                    let args_dict = PyDict::new_bound(py);
                                    args_dict.set_item("index", index)?;
                                    let func_args_dict = PyDict::new_bound(py);
                                    func_args_dict.set_item("arguments", arguments)?;
                                    args_dict.set_item("function", func_args_dict)?;

                                    results.append(("tool_call_args", args_dict))?;
                                }

                                self.emitted_calls += 1;
                                let after_match = &self.buf[end_idx + "<|tool_call_end|>".len()..];
                                self.buf = after_match.to_string();
                                continue;
                            }
                        }
                    }
                }
            }
            break;
        }
        Ok(())
    }
}
