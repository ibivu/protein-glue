using GZip

CHUNK_LENGTH = 1024 * 1024 * 128

function write_to_file(accession::String, data::Vector{UInt8})
    GZip.open("output/pfam_split/$(accession).shm.gz", "w") do f
        num_chunks  = Int64(ceil(length(data) / CHUNK_LENGTH))
        for n = 1:num_chunks
            begin_ = ((n - 1) * CHUNK_LENGTH) + 1
            if n === num_chunks
                write(f, data[begin_:end])
            else
                write(f, data[begin_:(n * CHUNK_LENGTH)])
            end
        end
    end
end

GZip.open("input/Pfam-A.rp55.gz") do f
    cur_data::IOBuffer = IOBuffer()
    cur_accession::Union{String, Nothing} = nothing

    for line in eachline(f, keep=true)
        if startswith(line, "# STOCKHOLM 1.0")
            data = take!(cur_data)
            if cur_accession !== nothing
                write_to_file(cur_accession, data)
            end
            cur_accession = nothing
        end
        if startswith(line, "#=GF AC")
            cur_accession = split(line)[3]
        end
        write(cur_data, line)
    end

    if cur_accession !== nothing
        write_to_file(cur_accession, take!(cur_data))
    end
end