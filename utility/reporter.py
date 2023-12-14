from datetime import datetime
import os

class Reporter(object):
    def __init__(self, ref_file: str, bhar_username: str, bhar_name: str, report_file_name: str = None, time_stats_file_name: str = None, output_directory: str = "output"):
        self.mode = 'a+'
        self.ref_file = ref_file
        self.bhar_name = bhar_name
        self.output_directory = output_directory
        self.start_time = datetime.now()
        self.previous_step_time = None

        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)

        now = str(datetime.now()).split('.')[0].replace(' ', '_').replace(":", "")
        if report_file_name is None:
            report_file_name = '{}_BHAR_baseline_{}.txt'.format(bhar_username, now)
        self.report_file = os.path.join(output_directory, report_file_name)

        if time_stats_file_name is None:
            time_stats_file_name = 'time_stats_{}.txt'.format(now)
        self.time_stats_file = os.path.join(output_directory, time_stats_file_name)

        with open(self.report_file, self.mode) as file:
            file.write('BHAR Analysis {}\n'.format(str(self.start_time).split('.')[0]))
            file.write('This analysis refers to {} file and to BHAR named {}\n\n'.format(os.path.basename(self.ref_file), self.bhar_name))

    def get_output_file(self):
        return self.report_file

    def get_dataset_name(self):
        if self.ref_file.endswith('/'):
            return str(os.path.basename(self.ref_file[:-1]))
        else:
            return str(os.path.basename(self.ref_file)).split('.')[0]

    def quantify_time_step(self):
        step_time = datetime.now()
        if self.previous_step_time is None:
            delta_min = str(step_time - self.start_time)
        else:
            delta_min = str(step_time - self.previous_step_time)

        self.previous_step_time = step_time
        return delta_min

    def total_elapsed_time(self):
        return str(datetime.now() - self.start_time)

    def write_title(self, title: str, print_elapsed_time: bool = False):
        with open(self.report_file, self.mode) as file:
            if print_elapsed_time:
                file.write('--> Time elapsed for this step: {}\n\n{}\n'.format(self.quantify_time_step(), title))
            else:
                file.write('{}\n'.format(title))

    def write_section(self, section_name: str, print_elapsed_time: bool = True):
        with open(self.report_file, self.mode) as file:
            if print_elapsed_time:
                file.write('--> Time elapsed for this step: {}\n\n{}'.format(
                    self.quantify_time_step(), section_name)
                )
            else:
                file.write('{}\n'.format(section_name))

    def write_subsection(self, subsection_name: str):
        with open(self.report_file, self.mode) as file:
            file.write('\n    {}\n'.format(subsection_name))

    def write_body(self, body: str):
        with open(self.report_file, self.mode) as file:
            file.write('        |{}\n'.format(body))

    def write_end(self):
        self.write_section('Conclusion')
        self.write_subsection('Total elapsed time for this analysis: {}'.format(self.total_elapsed_time()))
        self.write_subsection('Thank you for using BHAR!')

    def export_data(self, stage, df, write=False):
        if write:
            now = str(datetime.now()).split('.')[0].replace(' ', '_').replace(":", "")
            filename = os.path.join(self.output_directory, f"data_after_{stage}_{now}.txt")
            for df_i in df:
                df_i.to_csv(filename, index=False, header=False, mode="a")
    
    def export_time_stats(self, section: str, elapsed_time: datetime, write: bool = True):
        if write:
            if os.path.exists(self.time_stats_file):
                with open(self.time_stats_file, "r") as file:
                    pre_time = datetime.strptime(file.readlines()[-1].split(" ")[-1].rstrip(), "%H:%M:%S.%f")
                    cummulative_time = str(pre_time + elapsed_time).split(" ")[-1]

                with open(self.time_stats_file, "a") as file:
                    file.write(f"Time elapsed for {section}: {elapsed_time}; {cummulative_time}\n")
            else:
                with open(self.time_stats_file, self.mode) as file:
                    file.write(f"Time elapsed for {section}: {elapsed_time}\n")
