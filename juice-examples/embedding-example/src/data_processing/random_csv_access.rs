use std::fs::File;
use std::io::*;

#[derive(Clone)]
/// Fast Access of Random File Lines
/// Stores byte offsets for each newline in a file and provides a method for retrieving a specific
/// line in O(bytes) time rather than O(lines*bytes).
///
/// Currently only designed to work with a CSV, but would theoretically work with any type of file
/// that provides byte data and is newline separable.
///
/// Does not provide any type of newline removal/exceptions within 'actual' lines, and assumes the
/// user has done this type of pre-processing. This is *not* in line with the CSV standard which does
/// allow for new lines within lines if they are quoted.
pub(crate) struct RandomFileAccess {
    input_file: String,
    line_offsets: Vec<usize>
}

impl RandomFileAccess {
    pub(crate) fn new(file_path: String) -> RandomFileAccess {
        let file = File::open(&file_path).unwrap();
        let mut buffer = BufReader::new(file);
        let mut current_offset = 0;
        let mut line_offsets: Vec<usize> = Vec::with_capacity(1000);
        loop {
            let (done, used) = {
                let available = match buffer.fill_buf() {
                    Ok(n) => n,
                    Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(_) => panic!("Unavailable"),
                };
                match memchr::memchr(b'\n', available) {
                    Some(i) => {
                        (true, i + 1)
                    }
                    None => {
                        (false, available.len())
                    }
                }
            };
            buffer.consume(used);
            current_offset += used;
            if done {
                line_offsets.push(current_offset);
            }
            if used == 0 {
                break
            }
        }

        RandomFileAccess {
            input_file: file_path,
            line_offsets
        }
    }
    pub(crate) fn read_from_line(&self, line: usize) -> Vec<u8> {
        let mut f = File::open(&self.input_file).unwrap();
        let byte_offset: usize = self.line_offsets[line+1] - self.line_offsets[line];
        let mut buffer = vec![0u8; byte_offset];
        f.seek(SeekFrom::Start(self.line_offsets[line] as u64)).unwrap();
        f.read(&mut buffer);
        buffer.into_iter().filter(|byte| *byte != 10).collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.line_offsets.len() - 1
    }

    pub(crate) fn empty(&self) -> bool {
        self.line_offsets.len() == 0
    }
}