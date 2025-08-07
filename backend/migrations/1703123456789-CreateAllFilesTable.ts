import { MigrationInterface, QueryRunner, Table } from "typeorm";

export class CreateAllFilesTable1703123456789 implements MigrationInterface {
    name = 'CreateAllFilesTable1703123456789'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.createTable(
            new Table({
                name: "all_files",
                columns: [
                    {
                        name: "id",
                        type: "uuid",
                        isPrimary: true,
                        isGenerated: true,
                        generationStrategy: "uuid",
                        default: "gen_random_uuid()"
                    },
                    {
                        name: "filename",
                        type: "varchar",
                        length: "255",
                        isNullable: false
                    },
                    {
                        name: "originalName",
                        type: "varchar",
                        length: "100",
                        isNullable: false
                    },
                    {
                        name: "mimeType",
                        type: "varchar",
                        length: "50",
                        isNullable: false
                    },
                    {
                        name: "size",
                        type: "int",
                        isNullable: false
                    },
                    {
                        name: "filePath",
                        type: "text",
                        isNullable: false
                    },
                    {
                        name: "description",
                        type: "varchar",
                        length: "255",
                        isNullable: true
                    },
                    {
                        name: "fileType",
                        type: "varchar",
                        length: "100",
                        default: "'pdf'",
                        isNullable: false
                    },
                    {
                        name: "createdAt",
                        type: "timestamp",
                        default: "CURRENT_TIMESTAMP",
                        isNullable: false
                    },
                    {
                        name: "updatedAt",
                        type: "timestamp",
                        default: "CURRENT_TIMESTAMP",
                        isNullable: false
                    }
                ]
            }),
            true
        );
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.dropTable("all_files");
    }
} 