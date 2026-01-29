package CoreGenomeMLST;

use Bio::KBase::AppService::AppConfig qw(application_backend_dir);
use File::Slurp;
use IPC::Run;
use Cwd qw(abs_path getcwd);
use File::Copy;
use File::Copy::Recursive qw(dircopy);
use File::Path qw(rmtree make_path);
use File::Spec;
use strict;
use Data::Dumper;
use File::Basename;
use File::Temp;
use JSON;
use Text::CSV qw(csv);
use Getopt::Long::Descriptive;
use P3DataAPI;
use YAML;
use Template;
use Module::Metadata;
use warnings;
use strict;

use base 'Class::Accessor';

__PACKAGE__->mk_accessors(qw(work_dir staging_dir output_dir app params));


sub new
{
    my($class) = @_;
    my $self = {};
    bless $self, $class;

    $self->{api} = P3DataAPI->new();
    
    return $self;
}

sub run_CoreGenomeMLST
{

    my($self, $app, $app_def, $raw_params, $params) = @_;
    my $begin_time = time();
    print STDERR "Processed parameters for application " . $app->app_definition->{id} . ": ", Dumper($params);
    #
    # Set up work and staging directories.
    # 
    $self->app($app);
    $self->params($params);

    my $tmp_dir = $params->{_tmpdir};
    $tmp_dir //= getcwd . "/tmp.$$";
    
    my $work_dir = "$tmp_dir/work";
    my $staging_dir = "$tmp_dir/staging";
    my $raw_fasta_dir = "$staging_dir/raw_fastas";
    my $clean_fasta_dir = "$tmp_dir/clean_fastas";
    my $output_dir = "$tmp_dir/output";

    $self->{work_dir} = $work_dir;
    $self->{staging_dir} = $staging_dir;
    $self->{raw_fasta_dir} = $raw_fasta_dir;
    $self->{clean_fasta_dir} = $clean_fasta_dir;
    $self->{output_dir} = $output_dir;

    eval {
        make_path($work_dir, $staging_dir, $raw_fasta_dir, $clean_fasta_dir, $output_dir);
        run($app, $params, $staging_dir, $output_dir, $work_dir, $raw_fasta_dir, $clean_fasta_dir);
        };
    my $err = $@;
    if ($err)
    {
	warn "Run failed: $@";
    }
    save_output_files($app, $output_dir);
    my $end_time = time();
    printf STDERR "TOTAL TIME ELAPSED %.2f\n", $end_time - $begin_time ;
}

sub run
{
    my($app, $params, $staging_dir, $output_dir, $work_dir, $raw_fasta_dir, $clean_fasta_dir) = @_;

    #
    # set up for genome groups
    #

    if ($params->{input_genome_type} ne 'genome_group' and
	$params->{input_genome_type} ne 'genome_fasta')
    {
	    die "Gene set type $params->{input_genome_type} not supported";
    }

    # FUTURE DEV NOTE: If we support other input options
    # The fastas need to end up in the raw_fasta_dir - NB March 2025

    ## NB DEV NB TO DO update to work with API change 
    #my $master_table_fp = "/home/nbowers/bvbrc-dev/dev_container/cgmlst_for_all/all_salmonella_inital_updated.tsv";
    my $master_table_dir = "/home/nbowers/bvbrc-dev/dev_container/cgmlst_for_all/rerunning_joins_11_25_2025/raw_genome_ids/";
    # nicole 
    # $config_vars{schema_location} = application_backend_dir . "/CoreGenomeMLST/chewbbaca_schemas";
    my $schema_name = lc($params->{input_schema_selection});
    #my $master_table_fp = $master_table_dir . "$params->{input_schema_selection}_11_25_2025_joined.tsv";
    my $master_table_fp = $master_table_dir . $schema_name . "_11_25_2025_joined.tsv";
    my $processed_id_fp = "$work_dir/processed_genome_ids.txt";
    my $unprocessed_id_fp = "$work_dir/unprocessed_genome_ids.txt";
    my @unprocessed_genome_ids;
    my $precomputed_clusters_dir = application_backend_dir . "/CoreGenomeMLST/precomputed_clusters/";
    my $precomputed_clusters_path = $precomputed_clusters_dir . lc($params->{input_schema_selection}) . ".cgMLSTv1.npz";

    if ($params->{input_genome_type} eq 'genome_group')
    {
        my $genome_group_path = $params->{input_genome_group};
        print($genome_group_path);
        my $group_name = basename($genome_group_path);
        my $api = P3DataAPI->new;
        my $group_genome_ids = $api->retrieve_patric_ids_from_genome_group($genome_group_path);
        my $genome_id_file = "$work_dir/genome_id_list.txt";
        open(my $fh, ">", $genome_id_file) or die "Could not open file: $!";
        foreach my $item (@$group_genome_ids) {  
            print $fh "$item\n";
        }
        close $fh;

        # search the precomputed profiles for the ids listed in the genome group
        open my $lfh, "<", $genome_id_file or die "Cannot open $genome_id_file: $!";
        my %list_ids;
        while (<$lfh>) {
            chomp;
            s/\r$//;      # remove CR if Windows line endings
            next unless $_;
            $list_ids{$_} = 0;   # initialize as "not seen yet"
        }
        close $lfh;

        # Scan master table.tsv (only column 1, FILE)
        open my $tfh, "<", $master_table_fp or die "Cannot open $master_table_fp: $!";
        my $header = <$tfh>;  # skip header
        while (<$tfh>) {
            chomp;
            my ($id) = split /\t/, $_, 2;
            if (exists $list_ids{$id}) {
                $list_ids{$id} = 1;   
            }
        }
        close $tfh;

        open my $ppg, ">", $processed_id_fp or die "Cannot write $processed_id_fp: $!";
        open my $upg, ">", $unprocessed_id_fp  or die "Cannot write $unprocessed_id_fp: $!";

        for my $id (sort keys %list_ids) {
            if ($list_ids{$id}) {
                print $ppg "$id\n";
            } else {
                print $upg "$id\n";
                push @unprocessed_genome_ids, $id;
            }
        }

        close $ppg;
        close $upg;

        # Edit this to retreive only the genome ids we need 
        $api ->retrieve_contigs_in_genomes(\@unprocessed_genome_ids, $raw_fasta_dir, "%s");
        # # my @genome_metadata_fields = (
        # #   "genome_name", "genome_id", "ncbi_taxon_id", "organism_name", "taxon_lineage_ids", "taxon_lineage_names",
        # #   "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species", "genome_status", "strain", 
        # #   "serovar", "isolation_source", "isolation_comments", "collection_date", "collection_year", "season", 
        # #   "isolation_country", "geographic_group", "geographic_country", "geographic_location", "host_name", "host_common_name",
        # #   "host_gender", "host_age", "lab_host", "passage", "sequencing_center", "sequencing_status", "sequencing_platform", "chromosomes", 
        # #   "plasmids", "contigs", "genome_length", "gc_content", "contig_l50", "contig_n50", "cds", "genome_quality", 
        # #   "antimicrobial_resistance", "antimicrobial_resistance_evidence", "bioproject_accession", "biosample_accession", "assembly_accession",
        # #   "sra_accession", "genbank_accession", "refseq_accession", "comments", "date_inserted", "date_modified", "public", "owner", "members", "",
        # #   "accession", "patric_id", "public", "genome_status", "state");
        my @genome_metadata_fields = (
        "genome_id", "genome_name", "species", "strain", "genbank_accessions", "subtype", "lineage", "clade", "host_group", 
        "host_common_name", "host_scientific_name", "collection_year", "geographic_group", "isolation_country", "genome_status", 
        "state_province", "state");
        # Get metadata for all genomes in the group
        my @genome_group_metadata = $api->retrieve_genome_metadata($group_genome_ids, \@genome_metadata_fields);
        my $json_string = encode_json(@genome_group_metadata);
        print $json_string;
        # write metdata to json
        my $top = getcwd;
        write_file("$top/genome_metadata.json", JSON::XS->new->pretty->canonical->encode(\@genome_group_metadata));
    }

    # Prep the config and get ready to run Snakemake
    print STDERR "Starting the config json....\n";
    my $json_string = encode_json($params);

    #
    # Create json config file for the execution of this workflow.
    # If we are in a production deployment, we can find the workflows
    # by looking in $KB_TOP/workflows/app-name
    # Otherwise they are in the module directory; this is indicated
    # by the value of $KB_MODULE_DIR (note this is set for both
    # deployed and dev-container builds; the deployment case
    # is determined by the existence of $KB_TOP/workflows)
    #

    my %config_vars;
    # temp

    my $wf_dir = "$ENV{KB_TOP}/workflows/$ENV{KB_MODULE_DIR}";
    if (! -d $wf_dir)
    {
	$wf_dir = "$ENV{KB_TOP}/modules/$ENV{KB_MODULE_DIR}/workflow";
    }
    -d $wf_dir or die "Workflow directory $wf_dir does not exist";

    #
    # Find snakemake. We need to put this in a standard location in the runtime but for now
    # use this.
    #
    my @sm_dirs = ("$ENV{KB_RUNTIME}/bin", "$ENV{KB_RUNTIME}/artic-ncov2019/bin");
    my $snakemake;
    for my $dir (@sm_dirs)
    {
	my $p = "$dir/snakemake";
	if (-x $p)
	{
	    $snakemake = $p;
	    last;
	}
    }
    if (!$snakemake)
    {
	die "Cannot find snakemake in @sm_dirs\n";
    }
    warn "Snakemake found at $snakemake\n";
    system($snakemake, "--version");

    $config_vars{cores} = $ENV{P3_ALLOCATED_CPU} // 2;
    $config_vars{schema_location} = application_backend_dir . "/CoreGenomeMLST/chewbbaca_schemas";
    $config_vars{precomputed_clusters_path} = $precomputed_clusters_path;
    $config_vars{master_tsv} = $master_table_fp;
    $config_vars{snakemake} = $snakemake;
    $config_vars{workflow_dir} = $wf_dir;
    $config_vars{input_data_dir} = $staging_dir;
    $config_vars{output_data_dir} = $output_dir;
    $config_vars{work_data_dir} = $work_dir;
    $config_vars{clean_data_dir} = $clean_fasta_dir;
    $config_vars{raw_fasta_dir} = $raw_fasta_dir;
    $config_vars{processed_genome_ids} = $processed_id_fp;
    $config_vars{unprocessed_genome_ids} = $unprocessed_id_fp;

    # add the params to the config file
    $config_vars{params} = $params;

    # write config to current working directory to avoid finding tmp dir
    my $top = getcwd;
    write_file("$top/config.json", JSON::XS->new->pretty->canonical->encode(\%config_vars));

    print STDERR "Check point 1: Starting snakemake....\n";

    my @cmd = (
        $snakemake,
        "--cores",
        $config_vars{cores},
        "--use-singularity",
        "--verbose",
        "--printshellcmds",
        "--keep-going",
        "--snakefile",
        # NB Dev
        # "$wf_dir/snakefile/run_cgMLST_snakefile"
        "/home/nbowers/bvbrc-dev/dev_container/modules/bvbrc_CoreGenomeMLST/workflow/snakefile/run_cgMLST_snakefile_precomputed_profiles"
    );
    print STDERR "Run: @cmd\n";

    my $ok = IPC::Run::run(\@cmd);
    if (!$ok)
    {
     die "Snakemake command failed $?: @cmd";
    }
}

#
# Run preflight to estimate size and duration.
#
sub preflight
{
    my($app, $app_def, $raw_params, $params) = @_;

    my $time = 60 * 60 * 12;
    my $pf = {
            	cpu => 1,
            	memory => "32G",
            	runtime => $time,
                storage => 0,
              };

    return $pf;
}

sub save_output_files
{
    my($app, $output) = @_;

    my %suffix_map = (
            tre => 'nwk',
            tsv => 'tsv',
            cgMLSTv1 => 'tsv',
            NJ => 'txt',
            ML => 'txt',
            vcf => 'vcf',
            fasta => 'contigs',
            html => 'html');
    my @suffix_map = map { ("--map-suffix", "$_=$suffix_map{$_}") } keys %suffix_map;

    if (opendir(D, $output))
    {
	while (my $p = readdir(D))
	{
	    next if ($p =~ /^\./);
	    my @cmd = ("p3-cp", "--recursive", @suffix_map, "$output/$p", "ws:" . $app->result_folder);
	    print STDERR "saving files to workspace... @cmd\n";
	    my $ok = IPC::Run::run(\@cmd);
	    if (!$ok)
	    {
		warn "Error $? copying output with @cmd\n";
	    }
	}
    closedir(D);
    }
}

1;