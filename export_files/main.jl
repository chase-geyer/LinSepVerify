using ArgParse

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--file", "-f"
            help = "Choose a file to execute"
            arg_type = String
            required = true
    end

    parsed_args = parse_args(s)

    if parsed_args["file"] == "file1"
        println("Executing file1...")
        # Add code to execute file1 here
    elseif parsed_args["file"] == "file2"
        println("Executing file2...")
        # Add code to execute file2 here
    else
        println("Invalid file choice. Please choose 'file1' or 'file2'.")
    end
end

main()